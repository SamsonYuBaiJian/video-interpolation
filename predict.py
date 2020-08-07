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
    parser.add_argument('--save_model_path', required=True)
    parser.add_argument('--first_image', required=True)
    parser.add_argument('--last_image', required=True)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(args.save_model_path, map_location=torch.device(device))

    flow = get_optical_flow(args.first_image, args.last_image)
    first = PIL.Image.open(args.first_image)
    last = PIL.Image.open(args.last_image)
    flow = PIL.Image.fromarray(flow)
    transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    ])
    first = transforms(first)
    last = transforms(last)
    flow = transforms(flow)

    model.eval()

    with torch.no_grad():
        img_recon = model(first.unsqueeze(0).to(device), last.unsqueeze(0).to(device), flow.unsqueeze(0).to(device))
        tensor_list = [first, img_recon.squeeze(0), last]
        imshow(torchvision.utils.make_grid(tensor_list))