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
    parser.add_argument('--saved_model_path', required=True)
    parser.add_argument('--first_image', required=True)
    parser.add_argument('--last_image', required=True)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    autoencoder = torch.load(args.saved_model_path, map_location=torch.device(device))

    flow = get_optical_flow(args.first_image, args.last_image)
    first = PIL.Image.open(args.first_image)
    last = PIL.Image.open(args.last_image)
    flow = PIL.Image.fromarray(flow)
    transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    first = transforms(first)
    last = transforms(last)
    flow = transforms(flow)

    autoencoder.eval()

    with torch.no_grad():
        img_recon = autoencoder(first.unsqueeze(0).to(device), last.unsqueeze(0).to(device), flow.unsqueeze(0).to(device))
        tensor_list = [first, img_recon.squeeze(0), last]
        imshow(torchvision.utils.make_grid(tensor_list))