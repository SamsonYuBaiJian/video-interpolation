import PIL
import torchvision.transforms as transforms
import numpy as np
import torch
import torchvision
import time
import cv2
import argparse
import os
from model import RRIN


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames_path', required=True)
    parser.add_argument('--save_vid_dir', required=True)
    parser.add_argument('--saved_model_path', required=True)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(args.saved_model_path, map_location=torch.device(device))
    model.eval()
    
    transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    os.makedirs(args.save_vid_dir, exist_ok=True)

    frames = sorted(os.listdir(args.frames_path))
    for i in range(len(frames)):
        if i % 2 == 0:
            first_path = os.path.join(args.frames_path, frames[i])
            last_path = os.path.join(args.frames_path, frames[i+2])
            first = PIL.Image.open(first_path)
            last = PIL.Image.open(last_path)

            first = transforms(first)
            last = transforms(last)

            with torch.no_grad():
                img_recon = model(first.unsqueeze(0).to(device), last.unsqueeze(0).to(device))
            
            img_recon = img_recon.squeeze(0)
            img_recon = img_recon.numpy().transpose((1, 2, 0))
            first = first.numpy().transpose((1, 2, 0))
            # last = last.numpy().transpose((1, 2, 0))

            PIL.Image.fromarray((first * 255).astype(np.uint8)).save("{}/{}.jpg".format(args.save_vid_dir, i))
            PIL.Image.fromarray((img_recon * 255).astype(np.uint8)).save("{}/{}.jpg".format(args.save_vid_dir, i + 1))
            # PIL.Image.fromarray((last * 255).astype(np.uint8)).save("{}/{}.jpg".format(args.save_vid_dir, i + 2))