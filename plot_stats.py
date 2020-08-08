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

    train_g_loss = [i[0] for i in stats['train_loss']]
    train_d_loss = [i[1] for i in stats['train_loss']]
    val_g_loss = [i[0] for i in stats['val_loss']]
    val_d_loss = [i[1] for i in stats['val_loss']]
    length = len(stats['train_loss'])
    epochs = [epoch_interval * i for i in range(length)]

    _, axes = plt.subplots(1, 2)
    axes[0].set_title('G Loss vs Epoch')
    axes[1].set_title('D Loss vs Epoch')
    axes[0].plot(epochs, train_g_loss, label='Train G loss')
    axes[0].plot(epochs, train_d_loss, label='Val G loss')
    axes[1].plot(epochs, val_g_loss, label='Train D loss')
    axes[1].plot(epochs, val_d_loss, label='Val D loss')

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', required=True)
    args = parser.parse_args()

    plot_stats(args.exp_dir)