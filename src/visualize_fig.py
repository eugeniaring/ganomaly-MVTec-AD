import numpy as np

import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,roc_curve

from src.utils import denormalization

def save_snapshot(x, mask, model, save_dir):
    model.eval()
    with torch.no_grad():
        x_fake_list = x
        recon, _, _,_ = model(x)

        test_img = x[0].cpu().numpy()
        test_img = denormalization(test_img)

        recon_img = recon[0].cpu().numpy()
        recon_img = denormalization(recon_img)  

        mask = mask[0].cpu().numpy().transpose(1, 2, 0).squeeze()      

        fig_img, ax_img = plt.subplots(1, 3, figsize=(12, 4))
        fig_img.subplots_adjust(bottom=0.2, wspace=0.2)

        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)

        ax_img[0].imshow(test_img)
        ax_img[0].title.set_text('Image')

        ax_img[1].imshow(mask, cmap='gray')
        ax_img[1].title.set_text('GroundTruth')

        ax_img[2].imshow(recon_img)
        ax_img[2].title.set_text('Reconst')

        fig_img.savefig(save_dir, dpi=100)
        fig_img.clf()
        plt.close(fig_img)

def eval_anomalies(scores,gt_list,params,save_dir):

        gt_list = np.asarray(gt_list)
        img_scores = scores.reshape(gt_list.shape)

        # calculate image ROCAUC
        fpr, tpr, _ = roc_curve(gt_list, img_scores)
        img_roc_auc = roc_auc_score(gt_list, img_scores)
        print('image ROCAUC: %.3f' % (img_roc_auc))
        plt.plot(fpr, tpr, label='%s img_ROCAUC: %.3f' % (params['obj'], img_roc_auc))
        plt.legend(loc="lower right")
        plt.savefig(save_dir, dpi=100)
