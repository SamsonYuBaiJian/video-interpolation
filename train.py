from model import Net, Discriminator
import torch
from torch.utils.data import DataLoader
from dataloader import VimeoDataset
import numpy as np
import argparse
from datetime import datetime
import os
import time
import pickle


def save_stats(save_dir, exp_time, hyperparams, stats):
    """
    Save the hyperparameter settings and per-epoch statistics for the experiment.

    Args:
       save_dir (string): The directory of the folder to create save the hyperparameter settings and training progress statistics.
       exp_type (string): Starting date and time of the experiment, to use as a unique string to name the statistics folder.
       hyperparams (dict): Hyperparameter settings of the experiment.
       stats (dict): Statistics of the experiment.
    """
    save_path = os.path.join(save_dir, exp_time)
    os.makedirs(save_path, exist_ok=True)
    if not os.path.exists(os.path.join(save_path, 'hyperparams.pickle')):
        with open(os.path.join(save_path, 'hyperparams.pickle'), 'wb') as handle:
            pickle.dump(hyperparams, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()
    with open(os.path.join(save_path, 'stats.pickle'), 'wb') as handle:
        pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default=20, type=int, help="number of epochs to train for")
    parser.add_argument('--lr', default=1e-4, type=float, help="learning rate")
    parser.add_argument('--batch_size', default=4, type=int, help="batch size")
    parser.add_argument('--vimeo_90k_path', type=str, required=True, help="path to your Vimeo-90k folder")
    parser.add_argument('--save_stats_path', type=str, required=True, help="path where you want to save your experiment details")
    parser.add_argument('--eval_every', default=1, type=int, help="how often to evaluate on the validation set")
    parser.add_argument('--max_num_images', default=None, help="specify if you do not want to train on all images")
    parser.add_argument('--save_model_path', required=True, type=str, help="path to save your trained weights, should end with .pt")
    parser.add_argument('--time_it', action='store_true', help="whether you want to print estimated timings")
    parser.add_argument('--time_check_every', default=20, type=int, help="how often you want to print estimated timings, by number of batches")
    args = parser.parse_args()

    # process information to save statistics
    exp_time = datetime.now().strftime("date%d%m%Ytime%H%M%S")
    hyperparams = {
        'num_epochs': args.num_epochs,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'eval_every': args.eval_every,
        'max_num_images': args.max_num_images
    }

    # instantiate models, loss functions and optimisers
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Net()
    model = model.to(device)
    discriminator = Discriminator()
    discriminator.weight_init(mean=0.0, std=0.02)
    discriminator = discriminator.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    d_optimizer = torch.optim.Adam(params=discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    mse_loss = torch.nn.MSELoss()
    mse_loss.to(device)
    bce_loss = torch.nn.BCELoss()
    bce_loss.to(device)

    # to store evaluation metrics
    train_loss = []
    val_loss = []
    current_best_val_loss = float('inf')

    # build dataloaders
    print('Building train/val dataloaders...')
    seq_dir = os.path.join(args.vimeo_90k_path, 'sequences')
    train_txt = os.path.join(args.vimeo_90k_path, 'tri_trainlist.txt')
    trainset = VimeoDataset(video_dir=seq_dir, text_split=train_txt)
    # 80:20 train/val split
    n = len(trainset)
    n_train = int(n * 0.8)
    n_val = n - n_train
    # fix the generator for reproducible results
    trainset, valset = torch.utils.data.random_split(trainset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print('Train/val dataloaders successfully built!')

    # start training
    print('\nTraining...')
    for epoch in range(args.num_epochs):
        num_batches = 0
        train_loss_epoch = [0, 0, 0]
        
        model.train()
        discriminator.train()

        # for time calculations
        start_time = time.time()
        if args.max_num_images is not None:
            train_batches = int(np.ceil(float(args.max_num_images) / args.batch_size))
        else:
            train_batches = len(trainloader)
        
        for i in trainloader:
            # load data
            first = i['first_last_frames'][0]
            last = i['first_last_frames'][1]
            mid = i['middle_frame']
            first, last, mid = first.to(device), last.to(device), mid.to(device)

            mid_recon, flow_t_0, flow_t_1, w1, w2 = model(first, last)

            # discriminator training
            d_optimizer.zero_grad()
            d_real_result = discriminator(first, mid, last)
            # prevent generator backward pass, to reduce computation time
            d_fake_result = discriminator(first, mid_recon.detach(), last)
            d_loss_real = 0.5 * bce_loss(d_real_result, torch.ones_like(d_real_result).to(device))
            d_loss_fake = 0.5 * bce_loss(d_fake_result, torch.zeros_like(d_fake_result).to(device))
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

            # custom RRIN training
            optimizer.zero_grad()
            d_fake_result = discriminator(first, mid_recon, last)
            loss =  0.999 * mse_loss(mid, mid_recon) + 0.001 * bce_loss(d_fake_result, torch.ones_like(d_fake_result).to(device))
            loss.backward()
            optimizer.step()

            # store stats       
            train_loss_epoch[0] += loss.item()
            train_loss_epoch[1] += d_loss_real.item()
            train_loss_epoch[2] += d_loss_fake.item()
            num_batches += 1

            if args.max_num_images is not None:
                if num_batches == train_batches:
                    break

            # train time calculations
            if args.time_it:
                time_now = time.time()
                time_taken = time_now - start_time
                start_time = time_now
                if num_batches == 1 or num_batches % args.time_check_every == 0:
                    batches_left = train_batches - num_batches
                    print('Epoch [{} / {}] Time per batch of {}: {} seconds --> {} seconds for {} / {} batches left, train loss: {}, d_loss_real: {}, d_loss_fake: {}'.format(
                        epoch+1, args.num_epochs, mid.shape[0], time_taken, time_taken * batches_left, batches_left, train_batches, loss.item(), d_loss_real.item(), d_loss_fake.item()))

        train_loss_epoch[0] /= num_batches
        train_loss_epoch[1] /= num_batches
        train_loss_epoch[2] /= num_batches
        print('Epoch [{} / {}] Train g_loss: {}, d_loss_real: {}, d_loss_fake: {}'.format(epoch+1, args.num_epochs, train_loss_epoch[0], train_loss_epoch[1], train_loss_epoch[2]))

        # for evaluation, save best model and statistics
        if epoch % args.eval_every == 0:
            train_loss.append([0,0,0])
            val_loss.append([0,0,0])
            train_loss[-1] = train_loss_epoch

            model.eval()
            discriminator.eval()

            start_time = time.time()
            val_batches = len(valloader)

            with torch.no_grad():
                num_batches = 0
                for i in valloader:
                    first = i['first_last_frames'][0]
                    last = i['first_last_frames'][1]
                    mid = i['middle_frame']
                    first, last, mid = first.to(device), last.to(device), mid.to(device)

                    mid_recon, _, _, _, _ = model(first, last)
                    d_fake_result = discriminator(first, mid_recon, last)
                    loss = g_loss = 0.999 * mse_loss(mid, mid_recon) + 0.001 * bce_loss(d_fake_result, torch.ones_like(d_fake_result).to(device))
                    d_real_result = discriminator(first, mid, last)
                    d_loss_real = 0.5 * bce_loss(d_real_result, torch.ones_like(d_real_result).to(device))
                    d_loss_fake = 0.5 * bce_loss(d_fake_result, torch.zeros_like(d_fake_result).to(device))

                    # store stats
                    val_loss[-1][0] += loss.item()
                    val_loss[-1][1] += d_loss_real.item()
                    val_loss[-1][2] += d_loss_fake.item()
                    num_batches += 1

                    # val time calculations
                    if args.time_it:
                        time_now = time.time()
                        time_taken = time_now - start_time
                        start_time = time_now
                        if num_batches == 1 or num_batches % args.time_check_every == 0:
                            batches_left = val_batches - num_batches
                            print('Evaluating at Epoch [{} / {}] {} seconds for {} / {} batches of {} left'.format(epoch+1, args.num_epochs, 
                                time_taken * batches_left, batches_left, val_batches, mid.shape[0]))

                val_loss[-1][0] /= num_batches
                val_loss[-1][1] /= num_batches
                val_loss[-1][2] /= num_batches
                print('Val g_loss: {}, d_loss_real: {}, d_loss_fake: {}'.format(val_loss[-1][0], val_loss[-1][1], val_loss[-1][2]))

                # save best model so far, according to validation loss
                if val_loss[-1][0] < current_best_val_loss:
                    current_best_val_loss = val_loss[-1][0]
                    torch.save(model, args.save_model_path)
                    print("Saved new best model!")

            # save statistics
            stats = {
                'train_loss': train_loss,
                'val_loss': val_loss,
            }
            save_stats(args.save_stats_path, exp_time, hyperparams, stats)
            print("Saved stats!")