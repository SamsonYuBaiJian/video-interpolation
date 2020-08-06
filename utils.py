import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import PIL
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2


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
    self.transform = transform
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
        if len(frames) == 4:
            self.first_last_frames.append([frames[1], frames[3], frames[0]])
            self.middle_frame.append(frames[2])

  def __len__(self):
      return len(self.first_last_frames)

  def __getitem__(self, idx):
    first_last_flow = [PIL.Image.open(self.first_last_frames_flow[idx][0]).convert("RGB"), PIL.Image.open(self.first_last_frames_flow[idx][1]).convert("RGB"), PIL.Image.open(self.first_last_frames_flow[idx][2]).convert("RGB")]
    mid = PIL.Image.open(self.middle_frame[idx]).convert("RGB")

    if self.transform:
      first_last_flow = [self.transform(first_last_flow[0]), self.transform(first_last_flow[1]), self.transform(first_last_flow[2])]
      mid = self.transform(mid)

    sample = {'first_last_frames_flow': first_last_flow, 'middle_frame': mid}

    return sample

# show middle frame with first and last frames
def imshow(inp):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    # mean = np.array(means)
    # std = np.array(stds)
    # inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show()

def generate(model, dataloader, num_images, device):
    was_training = model.training
    model.eval()
    cnt = 0
    for i in dataloader:
        first = i['first_last_frames_flow'][0].unsqueeze(0).to(device)
        last = i['first_last_frames_flow'][1].unsqueeze(0).to(device)
        flow = i['first_last_frames_flow'][2].unsqueeze(0).to(device)
        mid = i['middle_frame']
        with torch.no_grad():
            recon = model(first, last, flow)
        recon = recon[0].squeeze(0).to('cpu')
        first = first.squeeze(0).to('cpu')
        last = last.squeeze(0).to('cpu')
        out = torchvision.utils.make_grid([first, last, mid, recon])
        imshow(out)
        cnt += 1
        if cnt == num_images:
            break
    model.train(mode=was_training)