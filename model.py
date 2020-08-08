import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# class Encoder(nn.Module):
#     def __init__(self, c):
#         super(Encoder, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=9, out_channels=c, kernel_size=4, stride=2, padding=1) # out: c x 128 x 224
#         self.conv2 = nn.Conv2d(in_channels=c, out_channels=c*2, kernel_size=4, stride=2, padding=1) # out: 2c x 64 x 112
#         self.conv3 = nn.Conv2d(in_channels=c*2, out_channels=c*2, kernel_size=4, stride=2, padding=1) # out: 2c x 32 x 56
#         self.conv4 = nn.Conv2d(in_channels=c*2, out_channels=c*3, kernel_size=4, stride=2, padding=1) # out: 3c x 16 x 28
#         self.bn1 = nn.BatchNorm2d(c)
#         self.bn2 = nn.BatchNorm2d(c*2)
#         self.bn3 = nn.BatchNorm2d(c*2)
#         self.bn4 = nn.BatchNorm2d(c*3)
            
#     def forward(self, first, last, flow):
#         x = torch.cat([first, last, flow], 1)
#         econv1 = F.relu(self.bn1(self.conv1(x)))
#         econv2 = F.relu(self.bn2(self.conv2(econv1)))
#         econv3 = F.relu(self.bn3(self.conv3(econv2)))
#         latent = torch.tanh(self.bn4(self.conv4(econv3)))
#         return latent, econv1, econv2, econv3


# class Decoder(nn.Module):
#     def __init__(self, c):
#         super(Decoder, self).__init__()
#         self.c = c
#         self.conv4 = nn.ConvTranspose2d(in_channels=c*3, out_channels=c*2, kernel_size=4, stride=2, padding=1)
#         self.conv3 = nn.ConvTranspose2d(in_channels=c*2*2, out_channels=c*2, kernel_size=4, stride=2, padding=1)
#         self.conv2 = nn.ConvTranspose2d(in_channels=c*2*2, out_channels=c, kernel_size=4, stride=2, padding=1)
#         self.conv1 = nn.ConvTranspose2d(in_channels=c*2, out_channels=3, kernel_size=4, stride=2, padding=1)
#         self.bn2 = nn.BatchNorm2d(c)
#         self.bn3 = nn.BatchNorm2d(c*2)
#         self.bn4 = nn.BatchNorm2d(c*2)
            
#     def forward(self, x, econv1, econv2, econv3):
#         x = F.relu(self.bn4(self.conv4(x)))
#         x = F.relu(self.bn3(self.conv3(torch.cat([x, econv3], dim=1))))
#         x = F.relu(self.bn2(self.conv2(torch.cat([x, econv2], dim=1))))
#         img = torch.tanh(self.conv1(torch.cat([x, econv1], dim=1)))
#         return img


class Encoder(nn.Module):
    def __init__(self, c, latent_dims):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=9, out_channels=c, kernel_size=4, stride=2, padding=1) # out: c x 128 x 224
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c*2, kernel_size=4, stride=2, padding=1) # out: 2c x 64 x 112
        self.conv3 = nn.Conv2d(in_channels=c*2, out_channels=c*2, kernel_size=4, stride=2, padding=1) # out: 2c x 32 x 56
        self.conv4 = nn.Conv2d(in_channels=c*2, out_channels=c*3, kernel_size=4, stride=2, padding=1) # out: 3c x 16 x 28
        self.bn1 = nn.BatchNorm2d(c)
        self.bn2 = nn.BatchNorm2d(c*2)
        self.bn3 = nn.BatchNorm2d(c*2)
        self.bn4 = nn.BatchNorm2d(c*3)
        self.fc = nn.Linear(c*3*16*28, latent_dims)
            
    def forward(self, first, last, flow):
        x = torch.cat([first, last, flow], 1)
        econv1 = F.relu(self.bn1(self.conv1(x)))
        econv2 = F.relu(self.bn2(self.conv2(econv1)))
        econv3 = F.relu(self.bn3(self.conv3(econv2)))
        econv4 = F.relu(self.bn4(self.conv4(econv3)))
        x = econv4.view(econv4.size(0), -1)
        latent = F.relu(self.fc(x))
        return latent, econv1, econv2, econv3, econv4


class Decoder(nn.Module):
    def __init__(self, c, latent_dims):
        super(Decoder, self).__init__()
        self.c = c
        self.conv4 = nn.ConvTranspose2d(in_channels=c*3*2, out_channels=c*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(in_channels=c*2*2, out_channels=c*2, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(in_channels=c*2*2, out_channels=c, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=c*2, out_channels=3, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(c)
        self.bn3 = nn.BatchNorm2d(c*2)
        self.bn4 = nn.BatchNorm2d(c*2)
        self.fc = nn.Linear(latent_dims, c*3*16*28)
            
    def forward(self, x, econv1, econv2, econv3, econv4):
        x = F.relu(self.fc)
        x = x.view(x.size(0), self.c*3, 16, 28)
        x = F.relu(self.bn4(self.conv4(torch.cat([x, econv4], dim=1))))
        x = F.relu(self.bn3(self.conv3(torch.cat([x, econv3], dim=1))))
        x = F.relu(self.bn2(self.conv2(torch.cat([x, econv2], dim=1))))
        img = torch.tanh(self.conv1(torch.cat([x, econv1], dim=1)))
        return img


class Autoencoder(nn.Module):
    def __init__(self, c, latent_dims):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(c, latent_dims)
        self.decoder = Decoder(c, latent_dims)
    
    def forward(self, first, last, flow):
        latent, econv1, econv2, econv3, econv4 = self.encoder(first, last, flow)
        x_recon = self.decoder(latent, econv1, econv2, econv3, econv4)
        return x_recon


# class Discriminator(nn.Module):
#     def __init__(self, c, latent_dims):
#         super(Discriminator, self).__init__()

#         self.model = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=c, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(c),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(in_channels=c, out_channels=c*2, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(c*2),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(in_channels=c*2, out_channels=c*3, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(c*3),
#             nn.LeakyReLU(0.2, inplace=True),
#         )
#         self.fc1 = nn.Linear(3*c*32*56, latent_dims)
#         self.fc2 = nn.Linear(latent_dims, 1)
#         self.sigmoid = nn.Sigmoid()


#     def forward(self, x):
#         x = self.model(x)
#         x = x.view(x.size(0), -1)
#         x = F.leaky_relu(self.fc1(x))
#         x = F.leaky_relu(self.fc2(x))
#         x = self.sigmoid(x)

#         return x