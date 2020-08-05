from model import VariationalAutoencoder, vae_loss
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from utils import VimeoDataset
import numpy as np
from utils import generate
import matplotlib.pyplot as plt


# parameters
num_epochs = 150
batch_size = 64
learning_rate = 1e-4
use_gpu = True
show_images_every = 10
eval_every = 10

vae = VariationalAutoencoder()

device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
vae = vae.to(device)

optimizer = torch.optim.Adam(params=vae.parameters(), lr=learning_rate, weight_decay=1e-5)

# set to training mode
vae.train()

train_loss_avg = []
train_psnr_avg = []
test_loss_avg = []
test_psnr_avg = []

trainset = VimeoDataset(video_dir='./sequences', text_split='./tri_trainlist.txt', transform= transforms.Compose([transforms.ToTensor()]))
testset = VimeoDataset(video_dir='./sequences', text_split='./tri_testlist.txt', transform= transforms.Compose([transforms.ToTensor()]))

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

print('Training ...')
for epoch in range(num_epochs):
    train_loss_avg.append(0)
    train_psnr_avg.append(0)
    num_batches = 0
    
    # use mini batches from trainloader
    for i in trainloader:
        first = i['first_last_frames'][0]
        last = i['first_last_frames'][1]
        mid = i['middle_frame']

        first, last, mid = first.to(device), last.to(device), mid.to(device)

        # vae reconstruction
        mid_recon, latent_mu, latent_logvar = vae(first, last)
        
        # reconstruction error
        loss = vae_loss(mid_recon, mid, latent_mu, latent_logvar)
        
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        
        # one step of the optmizer (using the gradients from backpropagation)
        optimizer.step()

        vae.eval()
        with torch.no_grad():
            # PSNR
            mid_recon = mid_recon.detach().to('cpu').numpy()
            mid = mid.detach().to('cpu').numpy()
            mse = (np.square(mid_recon - mid)).mean(axis=(1,2,3))
            psnr = 10 * np.log10(1 / mse)
            train_psnr_avg[-1] += np.mean(psnr)
        vae.train()
        
        train_loss_avg[-1] += loss.item()
        num_batches += 1

    train_loss_avg[-1] /= num_batches
    train_psnr_avg[-1] /= num_batches
    print('Epoch [%d / %d] average TRAIN reconstruction error: %f, TRAIN PSNR: %f' % (epoch+1, num_epochs, train_loss_avg[-1], train_psnr_avg[-1]))
    
    if (epoch+1) % show_images_every == 0:
        generate(vae, testset, 5, device)

    if (epoch+1) % eval_every == 0:
        test_loss_avg.append(0)
        test_psnr_avg.append(0)

        print('Evaluating...')
        vae.eval()
        with torch.no_grad():
            num_batches = 0
            for i in testloader:
                first = i['first_last_frames'][0]
                last = i['first_last_frames'][1]
                mid = i['middle_frame']

                first, last, mid = first.to(device), last.to(device), mid.to(device)

                # vae reconstruction
                mid_recon, latent_mu, latent_logvar = vae(first, last)
                
                # reconstruction error
                loss = vae_loss(mid_recon, mid, latent_mu, latent_logvar)

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
            print('Epoch [%d / %d] average TEST reconstruction error: %f, TEST PSNR: %f' % (epoch+1, num_epochs, test_loss_avg[-1], test_psnr_avg[-1]))
        vae.train()


# show how loss evolves during training
plt.ion()
fig = plt.figure()
plt.plot(train_loss_avg)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()