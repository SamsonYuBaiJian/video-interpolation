import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import PIL
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle


class VimeoDataset(Dataset):
    def __init__(self, video_dir, text_split, transform=None):
        """
        Args:
            video_dir (string): Vimeo-90k sequences directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.video_dir = video_dir
        self.text_split = text_split
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        self.middle_frame = []
        self.first_last_frames_flow = []

        with open(self.text_split, 'r') as f:
            filenames = f.readlines()
            f.close()
        final_filenames = []
        for i in filenames:
            final_filenames.append(os.path.join(self.video_dir, i.split('\n')[0]))

        for f in final_filenames:
            try:
                frames = [os.path.join(f, i) for i in os.listdir(f)]
            except:
                continue
            frames = sorted(frames)
            print(frames)
            if len(frames) == 4:
                self.first_last_frames_flow.append([frames[1], frames[3], frames[0]])
                self.middle_frame.append(frames[2])

    def __len__(self):
        return len(self.first_last_frames_flow)

    def __getitem__(self, idx):
        first_last_flow = [PIL.Image.open(self.first_last_frames_flow[idx][0]).convert("RGB"), PIL.Image.open(self.first_last_frames_flow[idx][1]).convert("RGB"), PIL.Image.open(self.first_last_frames_flow[idx][2]).convert("RGB")]
        mid = PIL.Image.open(self.middle_frame[idx]).convert("RGB")

        if self.transform:
            first_last_flow = [self.transform(first_last_flow[0]), self.transform(first_last_flow[1]), self.transform(first_last_flow[2])]
            mid = self.transform(mid)

        sample = {'first_last_frames_flow': first_last_flow, 'middle_frame': mid}

        return sample


def get_optical_flow(first, last):
    """
    Takes in image file paths, returns BGR numpy array.
    """
    first = cv2.imread(first)
    last = cv2.imread(last)
    hsv = np.zeros_like(first)
    first = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    hsv[...,1] = 255
    last = cv2.cvtColor(last, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(first, last, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    return bgr


def save_optical_flow(video_dir, text_split):
    """
    Args:
        video_dir (string): Vimeo-90k sequences directory.
    """
    with open(text_split, 'r') as f:
        filenames = f.readlines()
        f.close()
    final_filenames = []
    for i in filenames:
        final_filenames.append(os.path.join(video_dir, i.split('\n')[0]))

    for f in final_filenames:
        try:
            frames = [os.path.join(f, i) for i in os.listdir(f)]
        except:
            continue
        frames = sorted(frames)
        if len(frames) == 3:
            bgr = get_optical_flow(frames[0], frames[2])
            cv2.imwrite(os.path.join('/'.join(frames[0].split('/')[:-1]),'flow.png'), bgr)


def save_stats(save_dir, exp_time, hyperparams, stats):
    save_path = os.path.join(save_dir, exp_time)
    os.makedirs(save_path, exist_ok=True)
    if not os.path.exists(os.path.join(save_path, 'hyperparams.pickle')):
        with open(os.path.join(save_path, 'hyperparams.pickle'), 'wb') as handle:
            pickle.dump(hyperparams, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()
    with open(os.path.join(save_path, 'stats.pickle'), 'wb') as handle:
        pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()


def get_psnr(mid, mid_recon):
    """
    Returns PSNR value for two NumPy arrays.
    """
    with torch.no_grad():
        # mean over batches
        mse = (np.square(mid_recon - mid)).mean(axis=(1,2,3))
        psnr = 10 * np.log10(1 / mse)
        return np.mean(psnr)