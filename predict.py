import PIL
import torchvision.transforms as transforms
from utils import imshow
import numpy as np
import torch
import torchvision
import time
from utils import get_optical_flow
import cv2
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_model_path', default='./model.pt')
    parser.add_argument('--first_image')
    parser.add_argument('--last_image')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(args.save_model_path, map_location=torch.device(device))

    first = cv2.imread(args.first_image)
    last = cv2.imread(args.last_image)
    flow = get_optical_flow(first, last)
    first = PIL.Image.fromarray(first)
    last = PIL.Image.fromarray(last)
    flow = PIL.Image.fromarray(flow)
    first = transforms.Compose([transforms.ToTensor()])(first)
    last = transforms.Compose([transforms.ToTensor()])(last)
    flow = transforms.Compose([transforms.ToTensor()])(flow)

    model.eval()

    with torch.no_grad():
        # for i in range(0,400,50):
        #     # latent_mu, latent_logvar = vae.encoder(first, last)
        #     latent_mu, latent_logvar = model.encoder(first.unsqueeze(0).to(device), last.unsqueeze(0).to(device), flow.unsqueeze(0).to(device))
        #     # print(latent_mu.shape, latent_logvar.shape)
        #     latent = model.latent_sample(latent_mu, latent_logvar)
        #     tensor_list = []
        #     for j in range(-20,21,40):
        #         latent[:,i] = j / 10.
        #         image_recon = model.decoder(latent)
        #         image_recon = image_recon.squeeze(0).cpu()
        #         tensor_list.append(image_recon)
        #     imshow(torchvision.utils.make_grid(tensor_list))
        img_recon = model(first, last, flow)
        tensor_list = [first, img_recon, last]
        imshow(torchvision.utils.make_grid(tensor_list))