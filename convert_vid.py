import PIL
import torchvision.transforms as transforms
import numpy as np
import torch
import time
import cv2
import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vid_path', required=True)
    parser.add_argument('--save_vid_path', required=True)
    parser.add_argument('--saved_model_path', required=True)
    args = parser.parse_args()

    # set up video capture
    video_capture = cv2.VideoCapture(args.vid_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    success, image = video_capture.read()

    # set up video writer
    width, height = image.shape[1], image.shape[0]
    video_writer = cv2.VideoWriter('./project' + '.mp4', cv2.VideoWriter_fourcc(*'MP4V') , fps*2.0, (width, height))

    frame1 = image
    gen_frame2 = 0
    frame2 = 0

    # Write the first frame of the video
    video_writer.write(frame1)

    while success:  
        success,image = video_capture.read()
        frame2 = image

        gen_frame2 = frame1 # TODO: insert function that genertes frame 2

        frame1 = image

        video_writer.write(gen_frame2)
        video_writer.write(frame2)

    video_writer.release()
    video_capture.release()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(args.saved_model_path, map_location=torch.device(device))
    model.eval()
    
    transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    os.makedirs(args.save_vid_path, exist_ok=True)

    # frames = sorted(os.listdir(args.vid_path))
    # for i in range(len(frames)):
    #     if i % 2 == 0:
    #         first_path = os.path.join(args.frames_path, frames[i])
    #         last_path = os.path.join(args.frames_path, frames[i+2])
    #         first = PIL.Image.open(first_path)
    #         last = PIL.Image.open(last_path)

    #         first = transforms(first)
    #         last = transforms(last)

    #         with torch.no_grad():
    #             img_recon = model(first.unsqueeze(0).to(device), last.unsqueeze(0).to(device))
            
    #         img_recon = img_recon.squeeze(0)
    #         img_recon = img_recon.numpy().transpose((1, 2, 0))
    #         first = first.numpy().transpose((1, 2, 0))

    #         PIL.Image.fromarray((first * 255).astype(np.uint8)).save("{}/{}.jpg".format(args.save_vid_dir, i))
    #         PIL.Image.fromarray((img_recon * 255).astype(np.uint8)).save("{}/{}.jpg".format(args.save_vid_dir, i + 1))