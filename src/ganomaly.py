import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm


def create_ganomaly(img_shape, parameters):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    generator = Generator(img_shape,parameters['latent_dim'])
    discriminator = Discriminator(img_shape)
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    return generator,discriminator, device  

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)



'''architecture with dim size 64x64'''

class Encoder(nn.Module):
    def __init__(self, img_shape,latent_dim=100):
        super().__init__()
        n_features = 16
        self.model = nn.Sequential(
            # State (3x64x64)
            nn.Conv2d(in_channels=3, out_channels=n_features, kernel_size=4, stride=2, padding=1),
            #nn.BatchNorm2d(16, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
 
            # State (16x32x32)
            nn.Conv2d(in_channels=n_features, out_channels=n_features*2, kernel_size=4, stride=2, padding=1),
            #nn.Dropout(0.3),
            nn.BatchNorm2d(n_features*2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (32x16x16)
            nn.Conv2d(in_channels=n_features*2, out_channels=n_features*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(n_features*4, affine=True),
            #nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),

            # State (64x8x8)
            nn.Conv2d(in_channels=n_features*4, out_channels=n_features*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(n_features*8, affine=True),
            #nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),

            # State (128x4x4)
            nn.Conv2d(in_channels=n_features*8, out_channels=latent_dim, kernel_size=4, stride=1, padding=0),
            #output: 100x1x1
        )



    def forward(self, img):
        features = self.model(img)
        #print('encoder shape: ',features.size())
        #features = features.view(img.shape[0], -1)
        #features = features.view(features.shape[0], -1, 1, 1)
        return features

class Decoder(nn.Module):
    def __init__(self, img_shape, latent_dim=100):
        super().__init__()
        self.img_shape = img_shape
        n_features = 16
        self.model = nn.Sequential(
            # State (100x1x1)
            nn.ConvTranspose2d(in_channels=latent_dim, out_channels=n_features*8, kernel_size=4, stride=1, padding=0),
            #nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(True),

            # State (128x4x4)
            nn.ConvTranspose2d(in_channels=n_features*8, out_channels=n_features*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=n_features*4),
            #nn.Dropout(0.4),
            nn.LeakyReLU(True),

            # State (64x8x8)
            nn.ConvTranspose2d(in_channels=n_features*4, out_channels=n_features*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=n_features*2),
            #nn.Dropout(0.4),
            nn.LeakyReLU(True),

            # State (16x16x16)
            nn.ConvTranspose2d(in_channels=n_features*2, out_channels=n_features, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=n_features),
            #nn.Dropout(0.4),
            nn.LeakyReLU(True),

            # State (128x32x32)
            nn.ConvTranspose2d(in_channels=n_features, out_channels=3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
             )

    def forward(self, input):
        #input = input.view(input.shape[0], -1, 1, 1)
        #print('decoder shape: ',input.size())
        img = self.model(input)
        #print('decoder shape: ',img.size())
        return img.view(img.shape[0], *self.img_shape)

'''Architecture 256x256'''

class Encoder(nn.Module):
    def __init__(self, img_shape,latent_dim=100):
        super().__init__()
        n_features = 16
        self.model = nn.Sequential(
            # State (16x128x128)
            nn.Conv2d(in_channels=3, out_channels=n_features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
 
            # State (32x64x64)
            nn.Conv2d(in_channels=n_features, out_channels=n_features*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(n_features*2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (64x32x32)
            nn.Conv2d(in_channels=n_features*2, out_channels=n_features*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(n_features*4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (128x16x16)
            nn.Conv2d(in_channels=n_features*4, out_channels=n_features*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(n_features*8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x8x8)
            nn.Conv2d(in_channels=n_features*8, out_channels=n_features*16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(n_features*16, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
          
            # State (512x4x4)
            nn.Conv2d(in_channels=n_features*16, out_channels=n_features*32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(n_features*32, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State (100x1x1)
            nn.Conv2d(in_channels=n_features*32, out_channels=latent_dim, kernel_size=4, stride=1, padding=0)            
        )



    def forward(self, img):
        features = self.model(img)
        #print("encoder size: ",features.size())
        return features

class Decoder(nn.Module):
    def __init__(self, img_shape, latent_dim=100):
        super().__init__()
        self.img_shape = img_shape
        n_features = 16
        self.model = nn.Sequential(
            # State (512x4x4)
            nn.ConvTranspose2d(in_channels=latent_dim, out_channels=n_features*32, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(True), 

            # State (256x8x8)
            nn.ConvTranspose2d(in_channels=n_features*32, out_channels=n_features*16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=n_features*16),
            nn.LeakyReLU(True), 
            
            # State (128x16x16)
            nn.ConvTranspose2d(in_channels=n_features*16, out_channels=n_features*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=n_features*8),
            nn.LeakyReLU(True),

            # State (128x32x32)
            nn.ConvTranspose2d(in_channels=n_features*8, out_channels=n_features*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=n_features*4),
            nn.LeakyReLU(True),

            # State (64x64x64)
            nn.ConvTranspose2d(in_channels=n_features*4, out_channels=n_features*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=n_features*2),
            nn.LeakyReLU(True),

            # State (16x128x128)
            nn.ConvTranspose2d(in_channels=n_features*2, out_channels=n_features, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=n_features),
            nn.LeakyReLU(True),

            # State (128x32x32)
            nn.ConvTranspose2d(in_channels=n_features, out_channels=3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
             )

    def forward(self, input):
        img = self.model(input)
        return img.view(img.shape[0], *self.img_shape)

class Generator(nn.Module):
    def __init__(self,img_shape,latent_dim):
        super(Generator, self).__init__()
        self.encoder = Encoder(img_shape,latent_dim)
        self.decoder = Decoder(img_shape,latent_dim)
        self.encoder2 = Encoder(img_shape,latent_dim)

    def forward(self,x):
        z = self.encoder(x)
        x_hat =  self.decoder(z)
        z_o = self.encoder2(x_hat)
        return x_hat,z, z_o,None

class Discriminator(nn.Module):
    def __init__(self,img_shape):
        super(Discriminator, self).__init__()

        model = Encoder(img_shape,1)
        layers = list(model.model.children())
        self.features = nn.Sequential(*layers[:-1])
        self.classifier = nn.Sequential(layers[-1])
        self.classifier.add_module('Sigmoid', nn.Sigmoid())
        self.model = model

    def forward(self, x):

        features = self.features(x)
        features = features 
        classifier = self.classifier(features)
        classifier = classifier.view(-1,1).squeeze(1)
        return classifier, features

class Trainer_ganomaly():
    def __init__(self,gen,disc,device,params):
        self.gen = gen
        self.disc = disc
        self.device = device 
        
        vars(self).update(params)
        self.gen_optimizer = torch.optim.Adam(self.gen.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        self.disc_optimizer = torch.optim.Adam(self.disc.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        self.l_adv = nn.MSELoss()
        self.l_con = nn.L1Loss() 
        self.l_enc = nn.MSELoss() 
        self.l_bce = nn.BCELoss()

    def train_epoch(self,dataloader):
        L_adv_epoch_loss = 0.0
        L_con_epoch_loss = 0.0
        L_enc_epoch_loss = 0.0
        L_total_epoch_loss = 0.0
        disc_epoch_loss = 0.0
        dataSize = len(dataloader.dataset)

        for batch in tqdm(dataloader):
            x = batch[0]
            batch_size = x.size(0)
            x = x.to(self.device)

            label_real = torch.ones(batch_size).to(self.device)
            label_fake = torch.zeros(batch_size).to(self.device)

            # Update Discriminator
            self.disc_optimizer.zero_grad()
            D_real, _ = self.disc(x)
            disc_loss_real = self.l_bce(D_real, label_real)
            outputs, _, _,_ = self.gen(x)
            D_fake, _ = self.disc(outputs.detach())

            disc_loss_fake = self.l_bce(D_fake, label_fake)
            disc_loss = (disc_loss_fake + disc_loss_real) * 0.5
            disc_loss.backward()
            self.disc_optimizer.step()
            disc_epoch_loss += disc_loss.item() * batch_size   

            # Update Generator
            self.gen_optimizer.zero_grad()
            outputs, latent_in, latent_out, _ = self.gen(x)

            _, feature_fake = self.disc(outputs)
            _, feature_real = self.disc(x)
            adv_loss = self.l_adv(feature_fake, feature_real.detach())
            con_loss = self.l_con(outputs, x)
            enc_loss = self.l_enc(latent_out, latent_in)

            total_loss = self.lambda_adv * adv_loss + \
                            self.lambda_con * con_loss + \
                            self.lambda_enc * enc_loss
            total_loss.backward()
            
            L_adv_epoch_loss += adv_loss.item() * batch_size
            L_con_epoch_loss += con_loss.item() * batch_size
            L_enc_epoch_loss += enc_loss.item() * batch_size
            L_total_epoch_loss += total_loss.item() * batch_size  

            self.gen_optimizer.step()
         
        L_adv_epoch_loss /= dataSize
        L_con_epoch_loss /= dataSize
        L_enc_epoch_loss /= dataSize
        L_total_epoch_loss /= dataSize
        disc_epoch_loss /= dataSize                   

        metrics_epoch = {"contextual_loss_avg":L_con_epoch_loss,"adv_loss_avg":L_adv_epoch_loss,"enc_loss_avg":L_enc_epoch_loss,"disc_loss_avg":disc_epoch_loss,"tot_loss_avg":L_total_epoch_loss}

        return metrics_epoch 

    def val_epoch(self,dataloader,save_mask=False):

        con_losses,enc_losses = [],[]
        self.gt_mask_list = []
        self.gt_list = []
        self.gen.eval()
        self.disc.eval()
        for batch in tqdm(dataloader):
            x = batch[0].to(self.device) 
            if save_mask:
                y, mask = batch[1], batch[2]   
                self.gt_mask_list.extend(mask.cpu().numpy())
                self.gt_list.extend(y.cpu().numpy())
            with torch.no_grad():
                outputs, latent_in, latent_out,_ = self.gen(x)
                error = torch.mean(torch.pow((latent_in-latent_out), 2), dim=1)
            con_losses.append(self.l_con(outputs, x).item())
            enc_losses.extend(error.cpu().numpy())

        con_losses = np.asarray(con_losses)
        con_loss_avg = con_losses.mean()
        enc_losses = np.array(enc_losses)

        enc_losses = (enc_losses - np.min(enc_losses)) / (np.max(enc_losses) - np.min(enc_losses))
        enc_loss_avg = enc_losses.mean()
        metrics_epoch = {"contextual_loss_avg":con_loss_avg,"enc_loss_avg":enc_loss_avg,"contextual_losses":con_losses,"enc_losses":enc_losses}
        return metrics_epoch, x, outputs

    def evaluate_data(self,dataloader):
        metrics_epoch = self.val_epoch(dataloader,save_mask=True)
        return metrics_epoch



     