import torch
from torch.utils.data import DataLoader
from dataloader import VimeoDataset
import argparse
import os
import numpy as np
import skimage


def get_psnr(mid, mid_recon):
    """
    Returns PSNR sum for two NumPy arrays with size (batch_size, ...).
    """
    print(mid_recon, mid)
    with torch.no_grad():
        skimage.measure.compare_psnr(mid_recon, mid, data_range=None)
        mse = (np.square(mid_recon - mid)).mean(axis=(1,2,3))
        psnr = 10 * np.log10(1 / mse)
        print(psnr.shape)
        return np.mean(psnr)


def get_ssim(mid, mid_recon):
    """
    Returns SSIM sum for two NumPy arrays with size (batch_size, ...).
    """
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vimeo_90k_path', type=str, required=True)
    parser.add_argument('--saved_model_path', type=str, required=True)
    args = parser.parse_args()

    # load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(args.saved_model_path, map_location=torch.device(device))
    model = model.to(device)
    model.eval()

    # build test dataloader
    print('Building test dataloader...')
    seq_dir = os.path.join(args.vimeo_90k_path, 'sequences')
    test_txt = os.path.join(args.vimeo_90k_path, 'tri_testlist.txt')
    testset = VimeoDataset(video_dir=seq_dir, text_split=test_txt)
    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
    print('Test dataloader successfully built!')

    # get evaluation metrics PSNR and SSIM for test set
    print('\nTesting...')
    with torch.no_grad():
        psnr = 0
        ssim = 0
        num_samples = len(testloader)
        for i in testloader:
            for i in testloader:
                first = i['first_last_frames'][0]
                last = i['first_last_frames'][1]
                mid = i['middle_frame']
                first, last, mid = first.to(device), last.to(device), mid.to(device)

                mid_recon, _, _, _, _ = model(first, last)

                psnr += get_psnr(mid.detach().to('cpu').numpy(), mid_recon.detach().to('cpu').numpy())

        psnr /= num_samples
        print('Test set PSNR: {}, SSIM: {}'.format(psnr, 0))