import os
import yaml
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse

from src.utils import EarlyStop
from src.ganomaly import create_ganomaly, Trainer_ganomaly
from src.visualize_fig import eval_anomalies
from src.dataset import import_data
from src.visualize_fig import save_snapshot,eval_anomalies

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", default="bottle", type=str, help='object name')
    opt = parser.parse_args()
    f = open('params.yaml','rb')
    params = yaml.load(f, Loader=yaml.FullLoader)
    params['obj']=opt.obj

    f.close()
    train_dataset, train_loader, val_loader = import_data(params,is_train=True)
    _, test_loader = import_data(params,is_train=False)
    img_shape = train_dataset[0][0].shape
    gen,disc, device = create_ganomaly(img_shape,params)
    trainer = Trainer_ganomaly(gen,disc,device,params)

    n_epochs, early_stopping, lr_scheduler = params['num_epochs'], params['early_stopping'], params['lr_scheduler']
    if lr_scheduler:
        scheduler = ReduceLROnPlateau(trainer.gen_optimizer, mode='min', patience=3, min_lr=1e-6, factor=0.5)
    if early_stopping:
        early_stopping = EarlyStop(patience=5)

    x_test_fixed, _, mask_test_fixed = iter(test_loader).next()
    x_test_fixed = x_test_fixed.to(device)

    for epoch in range(n_epochs):
        train_met = trainer.train_epoch(train_loader)
        val_met, _, _ = trainer.val_epoch(val_loader)
        if early_stopping:
            early_stopping(val_met['loss'],trainer.gen,trainer.gen_optimizer)
            if early_stopping.early_stop:
                break
        if lr_scheduler:
            scheduler.step(val_met['loss'])
        print("\nEpoch: {}: train adv_loss: {:.3f} train con_loss: {:.3f} train enc_loss: {:.3f}  train tot loss: {:.3f}".format(epoch,train_met['adv_loss_avg'],train_met['contextual_loss_avg'],train_met['enc_loss_avg'],train_met['tot_loss_avg']))
        print("\nval con_loss: {:.3f} val enc_loss: {:.3f}".format(val_met['contextual_loss_avg'],val_met['enc_loss_avg']))

    save_sample = os.path.join(params['save_dir'], '{}.jpg'.format(params['obj']))
    if  os.path.exists(params['save_dir'])==False:
       os.makedirs(params['save_dir'])

    save_roc_auc = os.path.join(params['save_dir_rocauc'], '{}.jpg'.format(params['obj']))
    if  os.path.exists(params['save_dir_rocauc'])==False:
       os.makedirs(params['save_dir_rocauc'])

    save_snapshot(x_test_fixed,mask_test_fixed, gen, save_sample) 

    test_metrics_epoch,_,_ = trainer.evaluate_data(test_loader)

    eval_anomalies(test_metrics_epoch['enc_losses'],trainer.gt_list,params,save_roc_auc)


if __name__ == '__main__':
    main()



