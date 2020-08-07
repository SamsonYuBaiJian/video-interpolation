from model import Autoencoder
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from utils import VimeoDataset, generate
import numpy as np
import matplotlib.pyplot as plt
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--channels', default=64, type=int)
    parser.add_argument('--latent_dims', default=512, type=int)
    parser.add_argument('--num_epochs', default=150, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--use_gpu', default=True)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--vimeo_90k_path', type=str)
    # parser.add_argument('--show_images_every', default=10, type=int)
    parser.add_argument('--eval_every', default=10, type=int)
    parser.add_argument('--max_num_images', default=None)
    parser.add_argument('--save_model_path', default='./model.pt')
    args = parser.parse_args()

    model = Autoencoder(args.channels, args.latent_dims)
    device = torch.device("cuda:0" if args.use_gpu and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = torch.nn.MSELoss()
    criterion.to(device)
    model.train()

    train_loss_avg = []
    train_psnr_avg = []
    test_loss_avg = []
    test_psnr_avg = []

    print('Building dataloaders...')
    trainset = VimeoDataset(video_dir='../vimeo-90k/sequences', text_split='../vimeo-90k/tri_trainlist.txt', transform= transforms.Compose([transforms.ToTensor()]))
    testset = VimeoDataset(video_dir='../vimeo-90k/sequences', text_split='../vimeo-90k/tri_testlist.txt', transform= transforms.Compose([transforms.ToTensor()]))
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print('Dataloaders successfully built!')

    current_best_eval_psnr = 0
    print('\nTraining...')
    for epoch in range(args.num_epochs):
        train_loss_avg.append(0)
        train_psnr_avg.append(0)
        num_batches = 0
        
        # use mini batches from trainloader
        for i in trainloader:
            first = i['first_last_frames_flow'][0]
            last = i['first_last_frames_flow'][1]
            flow = i['first_last_frames_flow'][2]
            mid = i['middle_frame']

            first, last, flow, mid = first.to(device), last.to(device), flow.to(device), mid.to(device)

            mid_recon = model(first, last, flow)
            loss = criterion(mid, mid_recon)
            
            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                # PSNR
                mid_recon = mid_recon.detach().to('cpu').numpy()
                mid = mid.detach().to('cpu').numpy()
                mse = (np.square(mid_recon - mid)).mean(axis=(1,2,3))
                psnr = 10 * np.log10(1 / mse)
                train_psnr_avg[-1] += np.mean(psnr)
            model.train()
            
            train_loss_avg[-1] += loss.item()
            num_batches += 1

            if args.max_num_images is not None:
                if num_batches == int(float(args.max_num_images) / args.batch_size):
                    break

        train_loss_avg[-1] /= num_batches
        train_psnr_avg[-1] /= num_batches
        print('Train error: %f, Train PSNR: %f' % (train_loss_avg[-1], train_psnr_avg[-1]))

        if (epoch+1) % args.eval_every == 0:
            test_loss_avg.append(0)
            test_psnr_avg.append(0)

            print('Evaluating...')
            model.eval()
            with torch.no_grad():
                num_batches = 0
                for i in testloader:
                    first = i['first_last_frames_flow'][0]
                    last = i['first_last_frames_flow'][1]
                    flow = i['first_last_frames_flow'][2]
                    mid = i['middle_frame']

                    first, last, flow, mid = first.to(device), last.to(device), flow.to(device), mid.to(device)

                    mid_recon = model(first, last, flow)
                    loss = criterion(mid, mid_recon)

                    # PSNR
                    mid_recon = mid_recon.detach().to('cpu').numpy()
                    mid = mid.detach().to('cpu').numpy()
                    mse = (np.square(mid_recon - mid)).mean(axis=(1,2,3))
                    psnr = 10 * np.log10(1 / mse)
                    test_psnr_avg[-1] += np.mean(psnr)
                    test_loss_avg[-1] += loss.item()
                    num_batches += 1

                test_loss_avg[-1] /= num_batches
                test_psnr_avg[-1] /= num_batches
                print('Test error: %f, Test PSNR: %f' % (test_loss_avg[-1], test_psnr_avg[-1]))

            if test_psnr_avg[-1] > current_best_eval_psnr:
                print("Saving new best model...")
                current_best_eval_psnr = test_psnr_avg[-1]
                torch.save(model, args.save_model_path)
                print("Saved!")

            model.train()


    # show how loss evolves during training
    plt.ion()
    fig = plt.figure()
    plt.plot(train_loss_avg)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()