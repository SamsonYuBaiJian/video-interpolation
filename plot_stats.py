from utils import plot_stats
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', required=True)
    args = parser.parse_args()

    plot_stats(args.exp_dir)