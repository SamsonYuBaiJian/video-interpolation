import PIL
import torchvision.transforms as transforms
import numpy as np
import torch
import cv2
import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames_path', required=True)
    parser.add_argument('--saved_model_path', required=True)
    args = parser.parse_args()

    # load pretrained model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(args.saved_model_path, map_location=torch.device(device))
    model.eval()
    
    transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    # make sure the three frames to be predicted are in the right order
    frames = sorted([i for i in os.listdir(args.frames_path) if os.path.isfile(os.path.join(args.frames_path, i))])

    # process frames
    first_path = os.path.join(args.frames_path, frames[0])
    last_path = os.path.join(args.frames_path, frames[2])
    first = PIL.Image.open(first_path)
    last = PIL.Image.open(last_path)
    first = transforms(first)
    last = transforms(last)

    # create path to store generated visualisations
    save_path = os.path.join(args.frames_path, 'generated')
    os.makedirs(save_path, exist_ok=True)

    t = 0.9
    # model prediction
    with torch.no_grad():
        img_recon, flow_t_0, flow_t_1, w1, w2 = model(first.unsqueeze(0).to(device), last.unsqueeze(0).to(device), t)
    
    # save middle frame prediction
    img_recon = img_recon.squeeze(0).numpy().transpose((1, 2, 0)) * 255
    img_recon = img_recon.astype(np.uint8)
    PIL.Image.fromarray(img_recon).save("{}/predicted_{}.jpg".format(save_path, t))

    # save optical flows
    flow_t_0 = flow_t_0.squeeze(0).numpy().transpose((1, 2, 0))
    hsv_t_0 = np.zeros((flow_t_0.shape[0], flow_t_0.shape[1], 3), dtype=np.uint8)
    hsv_t_0[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow_t_0[..., 0], flow_t_0[..., 1])
    hsv_t_0[..., 0] = ang * 180 / np.pi / 2
    hsv_t_0[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr_t_0 = cv2.cvtColor(hsv_t_0, cv2.COLOR_HSV2BGR)
    PIL.Image.fromarray(bgr_t_0).save("{}/flow_t_0.jpg".format(save_path))
    flow_t_1 = flow_t_1.squeeze(0).numpy().transpose((1, 2, 0))
    hsv_t_1 = np.zeros((flow_t_1.shape[0], flow_t_1.shape[1], 3), dtype=np.uint8)
    hsv_t_1[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow_t_1[..., 0], flow_t_1[..., 1])
    hsv_t_1[..., 0] = ang * 180 / np.pi / 2
    hsv_t_1[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr_t_1 = cv2.cvtColor(hsv_t_1, cv2.COLOR_HSV2BGR)
    PIL.Image.fromarray(bgr_t_1).save("{}/flow_t_1.jpg".format(save_path))

    # save weight maps
    w1 = w1.squeeze().numpy()
    PIL.Image.fromarray((w1 * 255).astype(np.uint8), 'L').save("{}/weight_map_t_0.jpg".format(save_path))
    w2 = w2.squeeze().numpy()
    PIL.Image.fromarray((w2 * 255).astype(np.uint8), 'L').save("{}/weight_map_t_1.jpg".format(save_path))