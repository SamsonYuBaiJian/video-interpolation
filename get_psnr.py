from model import Autoencoder
import torch
from torch.utils.data import DataLoader, Dataset
from utils import VimeoDataset, get_psnr
import numpy as np
import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gpu', default=True)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--vimeo_90k_path', type=str, required=True)
    parser.add_argument('--saved_model_path', type=str, required=True)
    args = parser.parse_args()

    # instantiate setup
    device = torch.device("cuda:0" if args.use_gpu and torch.cuda.is_available() else "cpu")
    autoencoder = torch.load(args.saved_model_path, map_location=torch.device(device))
    autoencoder = autoencoder.to(device)
    autoencoder.eval()
    mse_loss = torch.nn.MSELoss()
    mse_loss.to(device)

    # build dataloaders
    print('Building test dataloader...')
    seq_dir = os.path.join(args.vimeo_90k_path, 'sequences')
    test_txt = os.path.join(args.vimeo_90k_path, 'tri_testlist.txt')
    testset = VimeoDataset(video_dir=seq_dir, text_split=test_txt)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print('Test dataloader successfully built!')

    # start training
    print('\nTesting...')
    with torch.no_grad():
        psnr = 0
        for i in testloader:
            num_batches = 0
            for i in testloader:
                first = i['first_last_frames_flow'][0]
                last = i['first_last_frames_flow'][1]
                flow = i['first_last_frames_flow'][2]
                mid = i['middle_frame']
                first, last, flow, mid = first.to(device), last.to(device), flow.to(device), mid.to(device)

                mid_recon = autoencoder(first, last, flow)

                psnr += get_psnr(mid.detach().to('cpu').numpy(), mid_recon.detach().to('cpu').numpy())
                num_batches += 1

        psnr /= num_batches
        print('Test set PSNR: {}'.format(psnr))