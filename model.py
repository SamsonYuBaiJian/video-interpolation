import torch
import torch.nn as nn
import torch.functional as F

capacity = 64
latent_dims = 512
variational_beta = 1

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        c = capacity
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=c, kernel_size=4, stride=2, padding=1) # out: c x 128 x 224
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c*2, kernel_size=4, stride=2, padding=1) # out: 2c x 64 x 112
        self.bn2 = nn.BatchNorm2d(2*c)
        self.conv3 = nn.Conv2d(in_channels=c*2, out_channels=c*2, kernel_size=4, stride=2, padding=1) # out: 2c x 32 x 56
        self.bn3 = nn.BatchNorm2d(2*c)
        self.conv4 = nn.Conv2d(in_channels=c*2, out_channels=c*3, kernel_size=4, stride=2, padding=1) # out: 3c x 16 x 28
        self.bn4 = nn.BatchNorm2d(3*c)
        self.fc_mu = nn.Linear(in_features=c*3*16*28, out_features=latent_dims)
        self.fc_logvar = nn.Linear(in_features=c*3*16*28, out_features=latent_dims)
            
    def forward(self, first, last):
        x = torch.cat([first, last], 1)
        econv1 = F.relu(self.bn1(self.conv1(x)))
        econv2 = F.relu(self.bn2(self.conv2(econv1)))
        econv3 = F.relu(self.bn3(self.conv3(econv2)))
        econv4 = F.relu(self.bn4(self.conv4(econv3)))
        x = econv4.view(econv4.size(0), -1) # flatten batch of multi-channel feature maps to a batch of feature vectors
        # notice, here we use use x for mu and for variance! 
        x_mu = self.fc_mu(x) 
        x_logvar = self.fc_logvar(x) #we don't calculate this from x_mu but from x!! This is crutial. 
        return x_mu, x_logvar, econv1, econv2, econv3, econv4

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        c = capacity
        self.fc = nn.Linear(in_features=latent_dims, out_features=c*3*16*28)
        self.bn4 = nn.BatchNorm2d(c*3)
        self.conv4 = nn.ConvTranspose2d(in_channels=c*3*2, out_channels=c*2, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(c*2)
        self.conv3 = nn.ConvTranspose2d(in_channels=c*2*2, out_channels=c*2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(c*2)
        self.conv2 = nn.ConvTranspose2d(in_channels=c*2*2, out_channels=c, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv1 = nn.ConvTranspose2d(in_channels=c*2, out_channels=3, kernel_size=4, stride=2, padding=1)
            
    def forward(self, x, econv1, econv2, econv3, econv4):
        x = self.fc(x)
        x = x.view(x.size(0), capacity*3, 16, 28) # unflatten batch of feature vectors to a batch of multi-channel feature maps
        x = self.bn4(x)
        dconv4 = F.relu(self.bn3(self.conv4(torch.cat([x, econv4], dim=1))))
        dconv3 = F.relu(self.bn2(self.conv3(torch.cat([dconv4, econv3], dim=1))))
        dconv2 = F.relu(self.bn1(self.conv2(torch.cat([dconv3, econv2], dim=1))))
        img = self.conv1(torch.cat([dconv2, econv1], dim=1))
        return img

class VariationalAutoencoder(nn.Module):
    def __init__(self):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, first, last):
        # remember our encoder output consists of x_mu and x_logvar
        latent_mu, latent_logvar, econv1, econv2, econv3, econv4 = self.encoder(first, last)
        # we sample from the distributions defined by mu and logvar
        # (function latent_sample defined below)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent, econv1, econv2, econv3, econv4)
        return x_recon, latent_mu, latent_logvar
    
    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_() #define normal distribution
            return eps.mul(std).add_(mu) #sample from normal distribution
        else:
            return mu

def vae_loss(recon_x, x, mu, logvar):
    # recon_x is the probability of a multivariate Bernoulli distribution p.
    # -log(p(x)) is then the pixel-wise binary cross-entropy.
    # Averaging or not averaging the binary cross-entropy over all pixels here
    # is a subtle detail with big effect on training, since it changes the weight
    # we need to pick for the other loss term by several orders of magnitude.
    # Not averaging is the direct implementation of the negative log likelihood,
    # but averaging makes the weight of the other loss term independent of the image resolution.
    recon_loss = F.mse_loss(recon_x.view(-1, 3*256*448), x.view(-1, 3*256*448), reduction='sum')
    
    # KL-divergence between the prior distribution over latent vectors
    # (the one we are going to sample from when generating new images)
    # and the distribution estimated by the generator for the given image.
    kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + variational_beta * kldivergence