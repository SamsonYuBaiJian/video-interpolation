import PIL
import torchvision.transforms as transforms
from utils import imshow
import numpy as np
import torch
import torchvision


test = PIL.Image.open('test.jpg')
test1 = PIL.Image.fromarray(np.asarray(test)[:,:800,:])
test2 = PIL.Image.fromarray(np.asarray(test)[:,800:,:])
test1 = transforms.Compose([transforms.Resize(448), transforms.CenterCrop((256,448)), transforms.ToTensor()])(test1)
test2 = transforms.Compose([transforms.Resize(448), transforms.CenterCrop((256,448)), transforms.ToTensor()])(test2)
imshow(test1)
imshow(test2)

vae.eval()

with torch.no_grad():
    for i in range(0,100,10):
        # latent_mu, latent_logvar = vae.encoder(first, last)
        latent_mu, latent_logvar = vae.encoder(test1.unsqueeze(0).to(device), test2.unsqueeze(0).to(device))
        tensor_list = []
        for j in range(-20,21,40):
            latent_mu[:,i] = j / 10.
            image_recon = vae.decoder(latent_mu)
            image_recon = image_recon.squeeze(0).cpu()
            tensor_list.append(image_recon)
        imshow(torchvision.utils.make_grid(tensor_list))

vae.train()