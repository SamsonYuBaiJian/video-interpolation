import PIL
import torchvision.transforms as transforms
from utils import imshow
import numpy as np
import torch
import torchvision
import time


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    vae = torch.load('./model.pt', map_location=torch.device('cpu'))
    test = PIL.Image.open('./test.jpg')
    test1 = PIL.Image.fromarray(np.asarray(test)[:,:800,:])
    test2 = PIL.Image.fromarray(np.asarray(test)[:,800:,:])
    test1 = transforms.Compose([transforms.Resize(448), transforms.CenterCrop((256,448)), transforms.ToTensor()])(test1)
    test2 = transforms.Compose([transforms.Resize(448), transforms.CenterCrop((256,448)), transforms.ToTensor()])(test2)

    vae.eval()

    with torch.no_grad():
        for i in range(0,400,50):
            # latent_mu, latent_logvar = vae.encoder(first, last)
            latent_mu, latent_logvar, econv1, econv2, econv3, econv4 = vae.encoder(test1.unsqueeze(0).to(device), test2.unsqueeze(0).to(device))
            latent = vae.latent_sample(latent_mu, latent_logvar)
            tensor_list = []
            for j in range(-20,21,40):
                latent[:,i:i+50] = j / 10.
                image_recon = vae.decoder(latent, econv1, econv2, econv3, econv4)
                image_recon = image_recon.squeeze(0).cpu()
                tensor_list.append(image_recon)
            imshow(torchvision.utils.make_grid(tensor_list))