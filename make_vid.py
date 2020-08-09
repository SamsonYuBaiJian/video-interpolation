import PIL
import torchvision.transforms as transforms
import numpy as np
import torch
import torchvision
import time
import cv2
import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vimeo_seq_dir', required=True)
    parser.add_argument('--save_vid_dir', required=True)
    parser.add_argument('--saved_model_path', required=True)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(args.saved_model_path, map_location=torch.device(device))
    model.eval()
    
    transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    sequences = sorted(os.listdir(args.vimeo_seq_dir))
    for i in range(len(sequences)):
        # if i % 2 == 0:
        path = os.path.join(args.vimeo_seq_dir, sequences[i])
        # flow = get_optical_flow(path + '/im1.png', path + '/im3.png')
        first = PIL.Image.open(path + '/im1.png')
        last = PIL.Image.open(path + '/im3.png')
        # flow = PIL.Image.fromarray(flow)

        first = transforms(first)
        last = transforms(last)
        # flow = transforms(flow)

        with torch.no_grad():
            img_recon = model(first.unsqueeze(0).to(device), last.unsqueeze(0).to(device))
        
        img_recon = img_recon.squeeze(0)
        img_recon = img_recon.numpy().transpose((1, 2, 0))
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        img_recon = std * img_recon + mean
        img_recon = np.clip(img_recon, 0, 1)

        # PIL.Image.fromarray((first.numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)).save("{}/{}.png".format(args.save_vid_dir, cnt))
        PIL.Image.fromarray((img_recon * 255).astype(np.uint8)).save("{}/{}.png".format(args.save_vid_dir, i))
        # PIL.Image.fromarray((last.numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)).save("{}/{}.png".format(args.save_vid_dir, i + 2))