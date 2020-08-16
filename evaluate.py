import torch
from torch.utils.data import DataLoader
from dataloader import VimeoDataset
import argparse
import os
import numpy as np
import skimage.metrics
import time


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vimeo_90k_path', type=str, required=True, help="path to your Vimeo-90k folder")
    parser.add_argument('--saved_model_path', type=str, required=True, help="path to your saved model weights")
    parser.add_argument('--time_check_every', type=int, default=20, help="how often you want to print estimated timings, by number of batches")
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
        start_time = time.time()
        cnt = 0

        for i in testloader:
            first = i['first_last_frames'][0]
            last = i['first_last_frames'][1]
            mid = i['middle_frame']
            first, last, mid = first.to(device), last.to(device), mid.to(device)

            mid_recon, _, _, _, _ = model(first, last)

            mid = mid.squeeze(0).detach().to('cpu').numpy().transpose((1, 2, 0))
            mid_recon = mid_recon.squeeze(0).detach().to('cpu').numpy().transpose((1, 2, 0))

            # calculate PSNR and SSIM
            psnr += skimage.metrics.peak_signal_noise_ratio(mid, mid_recon, data_range=1)
            ssim += skimage.metrics.structural_similarity(mid, mid_recon, data_range=1, multichannel=True)

            time_now = time.time()
            time_taken = time_now - start_time
            start_time = time_now

            cnt += 1

            if cnt == 1 or cnt % args.time_check_every == 0:
                samples_left = num_samples - cnt
                print('Time per sample: {} seconds --> {} seconds for {} / {} samples left'.format(time_taken, time_taken * samples_left, samples_left, num_samples))

        psnr /= num_samples
        ssim /= num_samples
        print('Test set PSNR: {}, SSIM: {}'.format(psnr, ssim))