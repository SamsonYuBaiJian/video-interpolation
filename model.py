import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Encoder(nn.Module):
    def __init__(self, c):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=9, out_channels=c, kernel_size=4, stride=2, padding=1) # out: c x 128 x 224
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c*2, kernel_size=4, stride=2, padding=1) # out: 2c x 64 x 112
        self.conv3 = nn.Conv2d(in_channels=c*2, out_channels=c*2, kernel_size=4, stride=2, padding=1) # out: 2c x 32 x 56
        self.conv4 = nn.Conv2d(in_channels=c*2, out_channels=c*3, kernel_size=4, stride=2, padding=1) # out: 3c x 16 x 28
            
    def forward(self, first, last, flow):
        x = torch.cat([first, last, flow], 1)
        econv1 = F.relu(self.conv1(x))
        econv2 = F.relu(self.conv2(econv1))
        econv3 = F.relu(self.conv3(econv2))
        latent = F.relu(self.conv4(econv3))
        # x = econv4.view(econv4.size(0), -1)
        return latent, econv1, econv2, econv3


class Decoder(nn.Module):
    def __init__(self, c):
        super(Decoder, self).__init__()
        self.c = c
        self.conv4 = nn.ConvTranspose2d(in_channels=c*3, out_channels=c*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(in_channels=c*2*2, out_channels=c*2, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(in_channels=c*2*2, out_channels=c, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=c*2, out_channels=3, kernel_size=4, stride=2, padding=1)
            
    # def forward(self, x):
    def forward(self, x, econv1, econv2, econv3):
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv3(torch.cat([x, econv3], dim=1)))
        x = F.relu(self.conv2(torch.cat([x, econv2], dim=1)))
        img = self.conv1(torch.cat([x, econv1], dim=1))
        return img


class Autoencoder(nn.Module):
    def __init__(self, c):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(c)
        self.decoder = Decoder(c)
    
    def forward(self, first, last, flow):
        latent, econv1, econv2, econv3 = self.encoder(first, last, flow)
        x_recon = self.decoder(latent, econv1, econv2, econv3)
        return x_recon


class Discriminator(nn.Module):
    def __init__(self, c):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=c, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)