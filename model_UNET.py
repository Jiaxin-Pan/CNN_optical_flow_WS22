import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from unet_parts import *

class UNet(pl.LightningModule):
    def __init__(self, hparams = None, train_set = None, val_set = None, logger = None, bilinear=True):
        super().__init__()
        self.hparams = hparams
        # set hyperparams
        self.train_set = train_set
        self.val_set = val_set
        self.logger = logger
        self.bilinear = bilinear

        self.inc = DoubleConv(self.hparams["n_channels"], 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, self.hparams["n_classes"])

    def forward(self, x):
        # the encoder for the previous image
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        #decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


    def general_step(self, batch, batch_idx, mode):
        images = batch
        layer = self.hparams["n_layers"]
        #prev_img = images[:,0].view(-1,3,128,128)
        #next_img = images[:,1].view(-1,3,128,128)
        prev_and_next_img = images[:,:layer].view(-1,self.hparams["n_channels"],128,128)
        motion_gt = images[:,layer].view(-1,3,128,128)
        #print(prev_img.shape)
        #idx_zeros = np.where(motion_gt!=0, 1, motion_gt)
        #idx_zeros = torch.tensor(idx_zeros).clone()#.detach()  
        #non_zeros = np.count_nonzero(idx_zeros)
        #print('non_zero',non_zeros)
        #motion_gt = motion_gt.double()
        
        # forward pass
        reconstruction = self.forward(prev_and_next_img)
        #reconstruction = torch.reshape(reconstruction, (-1,3, 128,128))
        #reconstruction = reconstruction.double()

        # loss
        loss_fn = nn.MSELoss()
        loss = loss_fn(reconstruction, motion_gt) #+ loss_fn(reconstruction*idx_zeros, motion_gt) *10
        
        reconstruction_np = reconstruction.detach().numpy()
        plt.imshow(np.squeeze(reconstruction_np[0].transpose(1,2,0)))
        plt.show()
        motion_gt_np = motion_gt.detach().numpy()
        plt.imshow(np.squeeze(motion_gt_np[0].transpose(1,2,0)))
        plt.show()
        
        return loss, reconstruction

    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
        return avg_loss

    def training_step(self, batch, batch_idx):
        loss, _ = self.general_step(batch, batch_idx, "train")
        tensorboard_logs = {'loss': loss}
        print("training loss", loss)
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss, _ = self.general_step(batch, batch_idx, "val")
        tensorboard_logs = {'validation_loss': loss}
        print("validation loss", loss)

        #self.logger.experiment.add_images(
        #    'reconstructions', images, self.current_epoch, dataformats='NCHW')
        return {'validation_loss': loss, 'log': tensorboard_logs}

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set, shuffle=True, batch_size=self.hparams['batch_size'])

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_set, batch_size=self.hparams['batch_size'])

    def configure_optimizers(self):
        
        
        optim = torch.optim.Adam([
                {'params': self.inc.parameters()},
                {'params': self.down1.parameters()},
                {'params': self.down2.parameters()},
                {'params': self.down3.parameters()},
                {'params': self.down4.parameters()},
                {'params': self.up1.parameters()},
                {'params': self.up2.parameters()},
                {'params': self.up3.parameters()},
                {'params': self.up4.parameters()},
                {'params': self.outc.parameters()},
            ], lr=self.hparams["learning_rate"])
        
        
        return optim


    """ Full assembly of the parts to form the complete network """



