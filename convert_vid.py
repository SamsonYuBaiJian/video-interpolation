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
    parser.add_argument('--print_every', default=150, help="specify the frame interval for printing progress")
    args = parser.parse_args()

    # set up model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(args.saved_model_path, map_location=torch.device(device))
    model.eval()

    # set up video capture
    video_capture = cv2.VideoCapture(args.vid_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    success, image = video_capture.read()

    # set up video writer
    width, height = image.shape[1], image.shape[0]
    video_writer = cv2.VideoWriter('{}.mp4'.format(args.save_vid_path), cv2.VideoWriter_fourcc(*'MP4V') , fps*2.0, (width, height))

    # check if width and height are divisible by 64, if not, padding is necessary to make inputs work with model's skip connections
    if width % 64 != 0:
        width_pad = int((np.floor(width / 64) + 1) * 64 - width)
    else:
        width_pad = 0
    if height % 64 != 0:
        height_pad = int((np.floor(height / 64) + 1) * 64 - height)
    else:
        height_pad = 0
    transforms = transforms.Compose([
        transforms.Pad((width_pad, height_pad, 0, 0)),
        transforms.ToTensor()
    ])

    # first frame
    frame1 = image
    # Write the first frame of the video
    video_writer.write(frame1)

    cnt = 1

    print("Starting video conversion, printing progress every {} frames...".format(args.print_every))
    while success:  
        success, image = video_capture.read()
        frame2 = image

        # do generation
        frame1_tensor = transforms(PIL.Image.fromarray(frame1))
        frame2_tensor = transforms(PIL.Image.fromarray(frame2))

        with torch.no_grad():
            gen_frame, _, _, _, _ = model(frame1_tensor.unsqueeze(0).to(device), frame2_tensor.unsqueeze(0).to(device))
        gen_frame = gen_frame.squeeze(0).cpu().numpy().transpose((1, 2, 0))
        gen_frame = (gen_frame * 255).astype(np.uint8)

        # get rid of padding for writing to video writer
        if width_pad > 0:
            gen_frame = gen_frame[:,width_pad:,:]
        if height_pad > 0:
            gen_frame = gen_frame[height_pad:,:,:]

        frame1 = image

        video_writer.write(gen_frame)
        video_writer.write(frame2)

        cnt += 1
        
        if cnt % args.print_every == 0:
            print('{} / {} frames left.'.format(frame_count - cnt, frame_count))
    
    print("Done!")

    video_writer.release()
    video_capture.release()