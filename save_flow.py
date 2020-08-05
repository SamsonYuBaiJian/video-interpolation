import cv2
import numpy as np
import os


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
            first = cv2.imread(frames[0])
            hsv = np.zeros_like(first)
            first = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
            hsv[...,1] = 255
            last = cv2.imread(frames[2])
            last = cv2.cvtColor(last, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(first, last, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
            cv2.imwrite(os.path.join('/'.join(frames[0].split('/')[:-1]),'flow.png'), bgr)
        break

if __name__ == '__main__':
    save_optical_flow('./vimeo-90k/vimeo_triplet/sequences', './vimeo-90k/vimeo_triplet/tri_trainlist.txt')