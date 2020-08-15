import argparse
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np


def plot_stats(exp_dir, save):
    """
    Plots the training progress graph.

    Args:
        exp_dir (string): Folder containing the pickle files for both the training statistics and hyperparameter settings.
        save (bool): Whether to save the graph automatically or not.
    """
    with open(os.path.join(exp_dir, 'stats.pickle'), 'rb') as handle:
        stats = pickle.load(handle)
        handle.close()
    with open(os.path.join(exp_dir, 'hyperparams.pickle'), 'rb') as handle:
        hyperparams = pickle.load(handle)
        handle.close()
    
    # prints the experiment's hyperparameter settings
    print("Experiment settings:\n{}".format(hyperparams))
    
    # load stats from saved pickle files
    epoch_interval = hyperparams['eval_every']

    train_g_loss = [i[0] * 100 for i in stats['train_loss']]
    train_d_loss_real = [i[1] for i in stats['train_loss']]
    train_d_loss_fake = [i[2] for i in stats['train_loss']]
    val_g_loss = [i[0] * 100 for i in stats['val_loss']]
    val_d_loss_real = [i[1] for i in stats['val_loss']]
    val_d_loss_fake = [i[2] for i in stats['val_loss']]
    length = len(stats['train_loss'])
    epochs = [epoch_interval * i + 1 for i in range(length)]
    print(val_g_loss)

    # plot stats
    _, axes = plt.subplots(1, 2)
    axes[0].set_title('Train Loss vs Epoch')
    axes[1].set_title('Val Loss vs Epoch')
    axes[0].plot(epochs, train_g_loss, label='Custom RRIN (x100)')
    axes[0].plot(epochs, train_d_loss_real, label='D - real')
    axes[0].plot(epochs, train_d_loss_fake, label='D - fake')
    axes[1].plot(epochs, val_g_loss, label='Custom RRIN (x100)')
    axes[1].plot(epochs, val_d_loss_real, label='D - real')
    axes[1].plot(epochs, val_d_loss_fake, label='D - fake')
    axes[0].set_xticks(np.arange(5,length,5))
    axes[1].set_xticks(np.arange(5,length,5))
    axes[0].legend()
    axes[1].legend()

    if save:
        plt.savefig('./plot_{}_{}.png'.format(hyperparams['lr'], hyperparams['batch_size']))

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', required=True)
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()

    plot_stats(args.exp_dir, args.save)