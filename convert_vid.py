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
    parser.add_argument('--vid_path', required=True, help="path to your input video")
    parser.add_argument('--save_vid_path', required=True, help="path to save your converted video")
    parser.add_argument('--saved_model_path', required=True, help="path to your saved model weights")
    args = parser.parse_args()

    # set up model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(args.saved_model_path, map_location=torch.device(device))
    model.eval()
    
    transforms = transforms.Compose([
        transforms.Pad((0,4,0,4)),
        transforms.ToTensor()
    ])

    # set up video capture
    video_capture = cv2.VideoCapture(args.vid_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    success, image = video_capture.read()

    # set up video writer
    width, height = image.shape[1], image.shape[0]
    video_writer = cv2.VideoWriter('{}.mp4'.format(args.save_vid_path), cv2.VideoWriter_fourcc(*'MP4V') , fps*2.0, (width, height))

    # first frame
    frame1 = image
    gen_frame2 = None
    frame2 = None

    # Write the first frame of the video
    video_writer.write(frame1)

    cnt = 0

    while success:  
        success, image = video_capture.read()
        frame2 = image

        # do generation
        frame1_tensor = transforms(frame1)
        frame2_tensor = transforms(frame2)
        with torch.no_grad():
            gen_frame, _, _, _, _ = model(frame1_tensor.unsqueeze(0).to(device), frame2_tensor.unsqueeze(0).to(device))
        gen_frame = gen_frame.squeeze(0).cpu().numpy().transpose((1, 2, 0))
        gen_frame = (gen_frame * 255).astype(np.uint8)

        frame1 = image

        video_writer.write(gen_frame)
        video_writer.write(frame2)

        cnt += 1
        print(cnt)

    video_writer.release()
    video_capture.release()