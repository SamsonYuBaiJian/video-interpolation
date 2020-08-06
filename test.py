import PIL
import torchvision.transforms as transforms
from utils import imshow
import numpy as np
import torch
import torchvision
import time
from utils import get_optical_flow


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    vae = torch.load('./model.pt', map_location=torch.device('cpu'))
    test = PIL.Image.open('./test.jpg')
    first = PIL.Image.fromarray(np.asarray(test)[:,:800,:])
    last = PIL.Image.fromarray(np.asarray(test)[:,800:,:])
    flow = get_optical_flow(first, last)
    first = transforms.Compose([transforms.Resize(448), transforms.CenterCrop((256,448)), transforms.ToTensor()])(first)
    last = transforms.Compose([transforms.Resize(448), transforms.CenterCrop((256,448)), transforms.ToTensor()])(last)
    flow = transforms.Compose([transforms.Resize(448), transforms.CenterCrop((256,448)), transforms.ToTensor()])(flow)

    vae.eval()

    with torch.no_grad():
        for i in range(0,400,50):
            # latent_mu, latent_logvar = vae.encoder(first, last)
            latent_mu, latent_logvar = vae.encoder(first.unsqueeze(0).to(device), last.unsqueeze(0).to(device), flow.unsqueeze(0).to(device))
            # print(latent_mu.shape, latent_logvar.shape)
            latent = vae.latent_sample(latent_mu, latent_logvar)
            tensor_list = []
            for j in range(-20,21,40):
                latent[:,i] = j / 10.
                image_recon = vae.decoder(latent)
                image_recon = image_recon.squeeze(0).cpu()
                tensor_list.append(image_recon)
            imshow(torchvision.utils.make_grid(tensor_list))