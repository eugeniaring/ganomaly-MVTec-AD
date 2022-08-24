import os
import yaml
from dataset import import_data
from ganomaly import create_ganomaly, Trainer_ganomaly, eval_anomalies
from tqdm import tqdm
import torch
from utils import EarlyStop, denorm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.utils import save_image

def save_snapshot(x, x2, model, save_dir, save_dir2):
    model.eval()
    with torch.no_grad():
        x_fake_list = x
        recon, _, _,_ = model(x)
        x_concat = torch.cat((x_fake_list, recon), dim=3)
        save_image(denorm(x_concat.data.cpu()), save_dir, nrow=1, padding=0)

        x_fake_list = x2
        recon, _, _,_ = model(x2)
        x_concat = torch.cat((x_fake_list, recon), dim=3)
        save_image(denorm(x_concat.data.cpu()), save_dir2, nrow=1, padding=0)

def main():
    import yaml
    f = open('params.yaml','rb')
    params = yaml.load(f, Loader=yaml.FullLoader)
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

    x_normal_fixed, _, _ = iter(val_loader).next()
    x_normal_fixed = x_normal_fixed.to(device)

    x_test_fixed, _, _ = iter(test_loader).next()
    x_test_fixed = x_test_fixed.to(device)

    for epoch in range(n_epochs):
        train_met = trainer.train_epoch(train_loader)
        val_met, _, _ = trainer.val_epoch(val_loader)
        if early_stopping:
            early_stopping(val_met['contextual_loss_avg'],trainer.gen,trainer.gen_optimizer)
            if early_stopping.early_stop:
                break
        if lr_scheduler:
            scheduler.step(val_met['contextual_loss_avg'])
        print("\nEpoch: {}: train adv_loss: {:.3f} train con_loss: {:.3f} train enc_loss: {:.3f}  train tot loss: {:.3f}".format(epoch,train_met['adv_loss_avg'],train_met['contextual_loss_avg'],train_met['enc_loss_avg'],train_met['tot_loss_avg']))
        print("\nval con_loss: {:.3f} val enc_loss: {:.3f}".format(val_met['contextual_loss_avg'],val_met['enc_loss_avg']))

    save_sample = os.path.join(params['save_dir'], '{}-val-images.jpg'.format(epoch))
    save_sample2 = os.path.join(params['save_dir'], '{}test-images.jpg'.format(epoch))
    if  os.path.exists(params['save_dir'])==False:
       os.makedirs(params['save_dir'])
    save_snapshot(x_normal_fixed, x_test_fixed, gen, save_sample, save_sample2) 

    test_metrics_epoch,test_imgs,recon_imgs = trainer.evaluate_data(test_loader)

    eval_anomalies(test_metrics_epoch['enc_losses'],test_metrics_epoch['enc_losses_imgs'],trainer.gt_list,trainer.gt_mask_list,test_imgs,recon_imgs,params)


if __name__ == '__main__':
    main()



