from model import Autoencoder, Vgg16_bn
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
    args = parser.parse_args()

    # process information to save statistics
    exp_time = datetime.now().strftime("date%d%m%Ytime%H%M%S")
    hyperparams = {
        'channels': args.channels,
        'num_epochs': args.num_epochs,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'eval_every': args.eval_every,
        'max_num_images': args.max_num_images
    }

    # instantiate setup
    device = torch.device("cuda:0" if args.use_gpu and torch.cuda.is_available() else "cpu")
    autoencoder = Autoencoder(args.channels)
    autoencoder = autoencoder.to(device)
    vgg16_bn = Vgg16_bn()
    vgg16_bn = vgg16_bn.to(device)
    optimizer = torch.optim.Adam(params=autoencoder.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = torch.nn.MSELoss()
    criterion.to(device)
    autoencoder.train()
    vgg16_bn.eval()

    # to store evaluation metrics
    train_loss = []
    train_psnr = []
    test_loss = []
    test_psnr = []
    current_best_eval_psnr = float('-inf')

    # build dataloaders
    print('Building dataloaders...')
    seq_dir = os.path.join(args.vimeo_90k_path, 'sequences')
    train_txt = os.path.join(args.vimeo_90k_path, 'tri_trainlist.txt')
    test_txt = os.path.join(args.vimeo_90k_path, 'tri_testlist.txt')
    trainset = VimeoDataset(video_dir=seq_dir, text_split=train_txt)
    testset = VimeoDataset(video_dir=seq_dir, text_split=test_txt)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print('Dataloaders successfully built!')

    # start training
    print('\nTraining...')
    for epoch in range(args.num_epochs):
        num_batches = 0
        train_psnr_epoch = 0
        train_loss_epoch = 0
        
        # use mini batches from trainloader
        for i in trainloader:
            # load data
            first = i['first_last_frames_flow'][0]
            last = i['first_last_frames_flow'][1]
            flow = i['first_last_frames_flow'][2]
            mid = i['middle_frame']

            # forward pass
            first, last, flow, mid = first.to(device), last.to(device), flow.to(device), mid.to(device)
            mid_recon = autoencoder(first, last, flow)
            loss = criterion(mid, mid_recon)
            mid_features = vgg16_bn(mid)
            mid_recon_features = vgg16_bn(mid_recon)
            # perceptual loss
            for i in range(len(mid_features)):
                loss += criterion(mid_features[i], mid_recon_features[i])
            
            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # store stats
            train_psnr_epoch += get_psnr(mid.detach().to('cpu').numpy(), mid_recon.detach().to('cpu').numpy())            
            train_loss_epoch += loss.item()
            num_batches += 1

            if args.max_num_images is not None:
                # print(np.ceil(float(args.max_num_images) / args.batch_size), num_batches)
                if num_batches == np.ceil(float(args.max_num_images) / args.batch_size):
                    break

        train_psnr_epoch /= num_batches
        train_loss_epoch /= num_batches
        print('Train error: %f, Train PSNR: %f' % (train_loss_epoch, train_psnr_epoch))

        # for evaluation, check test dataset, save best model and save statistics
        if epoch % args.eval_every == 0:
            train_loss.append(0)
            train_psnr.append(0)
            test_loss.append(0)
            test_psnr.append(0)
            train_loss[-1] += train_loss_epoch
            train_psnr[-1] += train_psnr_epoch

            # check test dataset
            print('Evaluating...')
            autoencoder.eval()
            with torch.no_grad():
                num_batches = 0
                # for i in testloader:
                for i in trainloader:
                    first = i['first_last_frames_flow'][0]
                    last = i['first_last_frames_flow'][1]
                    flow = i['first_last_frames_flow'][2]
                    mid = i['middle_frame']

                    first, last, flow, mid = first.to(device), last.to(device), flow.to(device), mid.to(device)

                    mid_recon = autoencoder(first, last, flow)
                    loss = criterion(mid, mid_recon)
                    mid_features = vgg16_bn(mid)
                    mid_recon_features = vgg16_bn(mid_recon)
                    # perceptual loss
                    for i in range(len(mid_features)):
                        loss += criterion(mid_features[i], mid_recon_features[i])

                    # store stats
                    test_psnr[-1] += get_psnr(mid.detach().to('cpu').numpy(), mid_recon.detach().to('cpu').numpy())
                    test_loss[-1] += loss.item()
                    num_batches += 1

                test_loss[-1] /= num_batches
                test_psnr[-1] /= num_batches
                print('Test error: %f, Test PSNR: %f' % (test_loss[-1], test_psnr[-1]))

            # save best model
            if test_psnr[-1] > current_best_eval_psnr:
                current_best_eval_psnr = test_psnr[-1]
                torch.save(autoencoder, args.save_model_path)
                print("Saved new best model!")

            # save statistics
            stats = {
                'train_loss': train_loss,
                'train_psnr': train_psnr,
                'test_loss': test_loss,
                'test_psnr': test_psnr
            }
            save_stats(args.save_stats_path, exp_time, hyperparams, stats)
            print("Saved stats!")

            autoencoder.train()