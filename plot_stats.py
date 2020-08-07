import argparse
import os
import pickle
import matplotlib.pyplot as plt


def plot_stats(exp_dir):
    with open(os.path.join(exp_dir, 'stats.pickle'), 'rb') as handle:
        stats = pickle.load(handle)
        handle.close()
    with open(os.path.join(exp_dir, 'hyperparams.pickle'), 'rb') as handle:
        hyperparams = pickle.load(handle)
        handle.close()
    
    print("Experiment settings:\n{}".format(hyperparams))
    
    # Plot stats
    epoch_interval = hyperparams['eval_every']

    train_loss = stats['train_loss']
    train_psnr = stats['train_psnr']
    val_loss = stats['test_loss']
    val_psnr = stats['test_psnr']
    test_loss = stats['test_loss']
    test_psnr = stats['test_psnr']
    length = len(train_loss)
    epochs = [epoch_interval * i for i in range(length)]

    _, axes = plt.subplots(1, 2)
    axes[0,0].set_title('Loss vs Epoch')
    axes[0,1].set_title('PSNR vs Epoch')
    axes[0,0].plot(epochs, train_loss, label='Train loss')
    axes[0,0].plot(epochs, val_loss, label='Val loss')
    axes[0,0].plot(epochs, test_loss, label='Test loss')
    axes[0,1].plot(epochs, train_psnr, label='Train PSNR')
    axes[0,1].plot(epochs, val_psnr, label='Val PSNR')
    axes[0,1].plot(epochs, test_psnr, label='Test PSNR')

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', required=True)
    args = parser.parse_args()

    plot_stats(args.exp_dir)