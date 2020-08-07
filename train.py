from model import Autoencoder, Discriminator
import torch
from torch.utils.data import DataLoader, Dataset
from utils import VimeoDataset, save_stats, get_psnr
import numpy as np
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import torchvision.models as models
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--channels', default=64, type=int)
    parser.add_argument('--num_epochs', default=150, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--use_gpu', default=True)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--vimeo_90k_path', type=str, required=True)
    parser.add_argument('--save_stats_path', type=str, required=True)
    parser.add_argument('--eval_every', default=10, type=int)
    parser.add_argument('--max_num_images', default=None)
    parser.add_argument('--save_model_path', default='./model.pt', required=True)
    parser.add_argument('--latent_dims', default=512, type=int)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    args = parser.parse_args()

    # process information to save statistics
    exp_time = datetime.now().strftime("date%d%m%Ytime%H%M%S")
    hyperparams = {
        'channels': args.channels,
        'num_epochs': args.num_epochs,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'eval_every': args.eval_every,
        'max_num_images': args.max_num_images,
        'latent_dims': args.latent_dims,
        'weight_decay': args.weight_decay
    }

    # instantiate setup
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    autoencoder = Autoencoder(args.channels)
    autoencoder = torch.nn.DataParallel(autoencoder, device_ids=[0,2,3])
    autoencoder = autoencoder.to(device)
    discriminator = Discriminator(args.channels, args.latent_dims)
    discriminator = torch.nn.DataParallel(discriminator, device_ids=[0,2,3])
    discriminator = discriminator.to(device)
    g_optimizer = torch.optim.Adam(params=autoencoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    d_optimizer = torch.optim.Adam(params=discriminator.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    mse_loss = torch.nn.MSELoss()
    mse_loss.to(device)
    bce_loss = torch.nn.BCELoss()
    bce_loss.to(device)

    # to store evaluation metrics
    train_loss = []
    train_psnr = []
    val_loss = []
    val_psnr = []
    test_loss = []
    test_psnr = []
    current_best_val_psnr = float('-inf')

    # build dataloaders
    print('Building dataloaders...')
    seq_dir = os.path.join(args.vimeo_90k_path, 'sequences')
    train_txt = os.path.join(args.vimeo_90k_path, 'tri_trainlist.txt')
    test_txt = os.path.join(args.vimeo_90k_path, 'tri_testlist.txt')
    trainset = VimeoDataset(video_dir=seq_dir, text_split=train_txt)
    # 80:20 train/val split
    n = len(trainset)
    n_train = int(n * 0.8)
    n_val = n - n_train
    # fix the generator for reproducible results
    trainset, valset = torch.utils.data.random_split(trainset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    testset = VimeoDataset(video_dir=seq_dir, text_split=test_txt)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print('Dataloaders successfully built!')

    # start training
    print('\nTraining...')
    for epoch in range(args.num_epochs):
        num_batches = 0
        train_psnr_epoch = 0
        train_loss_epoch = [0, 0]
        
        autoencoder.train()
        discriminator.train()
        for i in trainloader:
            # load data
            first = i['first_last_frames_flow'][0]
            last = i['first_last_frames_flow'][1]
            flow = i['first_last_frames_flow'][2]
            mid = i['middle_frame']
            first, last, flow, mid = first.to(device), last.to(device), flow.to(device), mid.to(device)
            valid = torch.ones(mid.shape[0], 1).to(device)
            fake = torch.zeros(mid.shape[0], 1).to(device)

            # autoencoder training
            g_optimizer.zero_grad()
            mid_recon = autoencoder(first, last, flow)
            g_loss = 0.999 * mse_loss(mid, mid_recon) + 0.001 * bce_loss(discriminator(mid_recon), valid)
            g_loss.backward()
            g_optimizer.step()

            # discriminator training
            d_optimizer.zero_grad()
            d_loss = 0.5 * (bce_loss(discriminator(mid), valid) + bce_loss(discriminator(mid_recon.detach()), fake))
            d_loss.backward()
            d_optimizer.step()

            # store stats
            train_psnr_epoch += get_psnr(mid.detach().to('cpu').numpy(), mid_recon.detach().to('cpu').numpy())            
            train_loss_epoch[0] = g_loss.item()
            train_loss_epoch[1] = d_loss.item()
            num_batches += 1

            if args.max_num_images is not None:
                if num_batches == np.ceil(float(args.max_num_images) / args.batch_size):
                    break

        train_psnr_epoch /= num_batches
        train_loss_epoch[0] /= num_batches
        train_loss_epoch[1] /= num_batches
        print('Epoch [{} / {}] Train error: {}, Train PSNR: {}'.format(epoch+1, args.num_epochs, train_loss_epoch, train_psnr_epoch))

        # for evaluation, save best model and statistics
        if epoch % args.eval_every == 0:
            train_loss.append([0,0])
            train_psnr.append(0)
            val_loss.append([0,0])
            val_psnr.append(0)
            test_loss.append([0,0])
            test_psnr.append(0)
            train_loss[-1] = train_loss_epoch
            train_psnr[-1] = train_psnr_epoch

            autoencoder.eval()
            discriminator.eval()

            # check val dataset
            print('Evaluating...')
            with torch.no_grad():
                num_batches = 0
                for i in valloader:
                    first = i['first_last_frames_flow'][0]
                    last = i['first_last_frames_flow'][1]
                    flow = i['first_last_frames_flow'][2]
                    mid = i['middle_frame']
                    first, last, flow, mid = first.to(device), last.to(device), flow.to(device), mid.to(device)
                    valid = torch.ones(mid.shape[0], 1).to(device)
                    fake = torch.zeros(mid.shape[0], 1).to(device)

                    mid_recon = autoencoder(first, last, flow)
                    g_loss = g_loss = 0.999 * mse_loss(mid, mid_recon) + 0.001 * bce_loss(discriminator(mid_recon), valid)
                    d_loss = 0.5 * (bce_loss(discriminator(mid), valid) + bce_loss(discriminator(mid_recon.detach()), fake))

                    # store stats
                    val_psnr[-1] += get_psnr(mid.detach().to('cpu').numpy(), mid_recon.detach().to('cpu').numpy())
                    val_loss[-1][0] += g_loss.item()
                    val_loss[-1][1] += d_loss.item()
                    num_batches += 1

                val_loss[-1][0] /= num_batches
                val_loss[-1][1] /= num_batches
                val_psnr[-1] /= num_batches
                print('Val error: {}, Val PSNR: {}'.format(val_loss[-1], val_psnr[-1]))

                # save best model
                if val_psnr[-1] > current_best_val_psnr:
                    current_best_val_psnr = val_psnr[-1]
                    torch.save(autoencoder, args.save_model_path)
                    print("Saved new best model!")

                # check val dataset
                print('Testing...')
                num_batches = 0
                for i in testloader:
                    first = i['first_last_frames_flow'][0]
                    last = i['first_last_frames_flow'][1]
                    flow = i['first_last_frames_flow'][2]
                    mid = i['middle_frame']
                    first, last, flow, mid = first.to(device), last.to(device), flow.to(device), mid.to(device)
                    valid = torch.ones(mid.shape[0], 1).to(device)
                    fake = torch.zeros(mid.shape[0], 1).to(device)

                    mid_recon = autoencoder(first, last, flow)
                    g_loss = g_loss = 0.999 * mse_loss(mid, mid_recon) + 0.001 * bce_loss(discriminator(mid_recon), valid)
                    d_loss = 0.5 * (bce_loss(discriminator(mid), valid) + bce_loss(discriminator(mid_recon.detach()), fake))

                    # store stats
                    test_psnr[-1] += get_psnr(mid.detach().to('cpu').numpy(), mid_recon.detach().to('cpu').numpy())
                    test_loss[-1][0] += g_loss.item()
                    test_loss[-1][1] += d_loss.item()
                    num_batches += 1

                test_loss[-1][0] /= num_batches
                test_loss[-1][1] /= num_batches
                test_psnr[-1] /= num_batches
                print('Test error: {}, Test PSNR: {}'.format(test_loss[-1], test_psnr[-1]))

            # save statistics
            stats = {
                'train_loss': train_loss,
                'train_psnr': train_psnr,
                'val_loss': val_loss,
                'val_psnr': val_psnr,
                'test_loss': test_loss,
                'test_psnr': test_psnr
            }
            save_stats(args.save_stats_path, exp_time, hyperparams, stats)
            print("Saved stats!")