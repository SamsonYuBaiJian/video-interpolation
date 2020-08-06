import torch
import torch.nn as nn
import torch.nn.functional as F


# variational_beta = 1

class Encoder(nn.Module):
    def __init__(self, c):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=9, out_channels=c, kernel_size=4, stride=2, padding=1) # out: c x 128 x 224
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c*2, kernel_size=4, stride=2, padding=1) # out: 2c x 64 x 112
        self.conv3 = nn.Conv2d(in_channels=c*2, out_channels=c*2, kernel_size=4, stride=2, padding=1) # out: 2c x 32 x 56
        self.conv4 = nn.Conv2d(in_channels=c*2, out_channels=c*3, kernel_size=4, stride=2, padding=1) # out: 3c x 16 x 28
        # self.fc_mu = nn.Linear(in_features=c*3*16*28, out_features=latent_dims)
        # self.fc_logvar = nn.Linear(in_features=c*3*16*28, out_features=latent_dims)
        # self.fc = nn.Linear(in_features=c*3*16*28, out_features=latent_dims)
            
    def forward(self, first, last, flow):
        x = torch.cat([first, last, flow], 1)
        econv1 = F.relu(self.conv1(x))
        econv2 = F.relu(self.conv2(econv1))
        econv3 = F.relu(self.conv3(econv2))
        latent = F.relu(self.conv4(econv3))
        # x = econv4.view(econv4.size(0), -1)
        # x_mu = self.fc_mu(x) 
        # x_logvar = self.fc_logvar(x)
        # return x_mu, x_logvar
        return latent, econv1, econv2, econv3 # econv4

class Decoder(nn.Module):
    def __init__(self, c):
        super(Decoder, self).__init__()
        self.c = c
        # self.fc = nn.Linear(in_features=latent_dims, out_features=c*3*16*28)
        self.conv4 = nn.ConvTranspose2d(in_channels=c*3, out_channels=c*2, kernel_size=4, stride=2, padding=1)
        # self.conv3 = nn.ConvTranspose2d(in_channels=c*2, out_channels=c*2, kernel_size=4, stride=2, padding=1)
        # self.conv2 = nn.ConvTranspose2d(in_channels=c*2, out_channels=c, kernel_size=4, stride=2, padding=1)
        # self.conv1 = nn.ConvTranspose2d(in_channels=c, out_channels=3, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(in_channels=c*2*2, out_channels=c*2, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(in_channels=c*2*2, out_channels=c, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=c*2, out_channels=3, kernel_size=4, stride=2, padding=1)
            
    # def forward(self, x):
    def forward(self, x, econv1, econv2, econv3):
        # x = self.fc(x)
        # x = x.view(x.size(0), self.c*3, 16, 28)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv3(torch.cat([x, econv3], dim=1)))
        x = F.relu(self.conv2(torch.cat([x, econv2], dim=1)))
        img = self.conv1(torch.cat([x, econv1], dim=1))
        return img

class Autoencoder(nn.Module):
    def __init__(self, c, latent_dims):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(c)
        self.decoder = Decoder(c)
    
    def forward(self, first, last, flow):
        # latent_mu, latent_logvar = self.encoder(first, last, flow)
        latent, econv1, econv2, econv3 = self.encoder(first, last, flow)
        # latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent, econv1, econv2, econv3)
        # return x_recon, latent_mu, latent_logvar
        return x_recon
    
    # def latent_sample(self, mu, logvar):
    #     if self.training:
    #         # the reparameterization trick
    #         std = logvar.mul(0.5).exp_()
    #         eps = torch.empty_like(std).normal_() #define normal distribution
    #         return eps.mul(std).add_(mu) #sample from normal distribution
    #     else:
    #         return mu

# def vae_loss(recon_x, x, mu, logvar):
#     recon_loss = F.mse_loss(recon_x.view(-1, 3*256*448), x.view(-1, 3*256*448), reduction='sum')
#     kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

#     return recon_loss + variational_beta * kldivergence