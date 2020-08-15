import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import PIL
import os


class VimeoDataset(Dataset):
    def __init__(self, video_dir, text_split, transform=None):
        """
        Dataset class for the Vimeo-90k dataset, available at http://toflow.csail.mit.edu/.

        Args:
            video_dir (string): Vimeo-90k sequences directory.
            text_split (string): Text file path in the Vimeo-90k folder, either `tri_trainlist.txt` or `tri_testlist.txt`.
            transform (callable, optional): Optional transform to be applied samples.
        """
        self.video_dir = video_dir
        self.text_split = text_split
        # default transform as per RRIN, convert images to tensors, with values between 0 and 1
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])
        self.middle_frame = []
        self.first_last_frames = []

        # open the given text file path that gives file names for train or test subsets
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
            # make sure images are in order, i.e. im1.png, im2.png, im3.png
            frames = sorted(frames)
            # make sure there are only 3 images in the Vimeo-90k triplet's folder for it to be a valid dataset sample
            if len(frames) == 3:
                self.first_last_frames.append([frames[0], frames[2]])
                self.middle_frame.append(frames[1])

    def __len__(self):
        return len(self.first_last_frames)

    def __getitem__(self, idx):
        first_last = [PIL.Image.open(self.first_last_frames[idx][0]).convert("RGB"), PIL.Image.open(self.first_last_frames[idx][1]).convert("RGB")]
        mid = PIL.Image.open(self.middle_frame[idx]).convert("RGB")

        if self.transform:
            first_last = [self.transform(first_last[0]), self.transform(first_last[1])]
            mid = self.transform(mid)

        sample = {'first_last_frames': first_last, 'middle_frame': mid}

        return sample