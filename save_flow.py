import cv2
import numpy as np
import os
from utils import get_optical_flow


def save_optical_flow(video_dir, text_split):
    with open(text_split, 'r') as f:
        filenames = f.readlines()
        f.close()
    final_filenames = []
    for i in filenames:
        final_filenames.append(os.path.join(video_dir, i.split('\n')[0]))

    for f in final_filenames:
        try:
            frames = [os.path.join(f, i) for i in os.listdir(f)]
        except:
            continue
        frames = sorted(frames)
        if len(frames) == 3:
            bgr = get_optical_flow(frames[0], frames[1])
            cv2.imwrite(os.path.join('/'.join(frames[0].split('/')[:-1]),'flow.png'), bgr)

if __name__ == '__main__':
    save_optical_flow('./vimeo-90k/vimeo_triplet/sequences', './vimeo-90k/vimeo_triplet/tri_trainlist.txt')
    save_optical_flow('./vimeo-90k/vimeo_triplet/sequences', './vimeo-90k/vimeo_triplet/tri_testlist.txt')