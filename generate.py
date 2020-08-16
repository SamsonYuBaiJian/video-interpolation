import PIL
import torchvision.transforms as transforms
import numpy as np
import torch
import cv2
import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames_path', required=True, help="directory containing the frames you want to interpolate with")
    parser.add_argument('--saved_model_path', required=True, help="path to your saved model weights")
    parser.add_argument('--t', default=0.5, type=float, metavar='[0-1]', help="timestep for interpolation")
    args = parser.parse_args()

    # load pretrained model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(args.saved_model_path, map_location=torch.device(device))
    model.eval()

    # make sure the two frames to be interpolated are in the right order
    frames = sorted([i for i in os.listdir(args.frames_path) if os.path.isfile(os.path.join(args.frames_path, i))])

    # process frames
    first_path = os.path.join(args.frames_path, frames[0])
    last_path = os.path.join(args.frames_path, frames[1])
    first = PIL.Image.open(first_path)
    last = PIL.Image.open(last_path)

    width, height = first.size

    # check if width and height are divisible by 64, if not, padding is necessary to make inputs work with skip connections
    if width % 64 != 0:
        width_pad = int((np.floor(width / 64) + 1) * 64 - width)
    else:
        width_pad = 0
    if height % 64 != 0:
        height_pad = int((np.floor(height / 64) + 1) * 64 - height)
    else:
        height_pad = 0
    transforms = transforms.Compose([
        transforms.Pad((width_pad, height_pad, 0, 0)),
        transforms.ToTensor()
    ])
    first = transforms(first)
    last = transforms(last)

    # create path to store generated visualisations
    save_path = os.path.join(args.frames_path, 'generated')
    os.makedirs(save_path, exist_ok=True)

    # model prediction
    with torch.no_grad():
        img_recon, flow_t_0, flow_t_1, w1, w2 = model(first.unsqueeze(0).to(device), last.unsqueeze(0).to(device), args.t)
    
    # save middle frame prediction
    img_recon = img_recon.squeeze(0).numpy().transpose((1, 2, 0)) * 255
    # get rid of padding after prediction
    if width_pad > 0:
        img_recon = img_recon[:,width_pad:,:]
    if height_pad > 0:
        img_recon = img_recon[height_pad:,:,:]
    img_recon = img_recon.astype(np.uint8)
    PIL.Image.fromarray(img_recon).save("{}/predicted_t={}.jpg".format(save_path, args.t))

    # save bidirectional optical flows
    flow_t_0 = flow_t_0.squeeze(0).numpy().transpose((1, 2, 0))
    # get rid of padding after prediction
    if width_pad > 0:
        flow_t_0 = flow_t_0[:,width_pad:,:]
    if height_pad > 0:
        flow_t_0 = flow_t_0[height_pad:,:,:]
    hsv_t_0 = np.zeros((flow_t_0.shape[0], flow_t_0.shape[1], 3), dtype=np.uint8)
    hsv_t_0[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow_t_0[..., 0], flow_t_0[..., 1])
    hsv_t_0[..., 0] = ang * 180 / np.pi / 2
    hsv_t_0[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr_t_0 = cv2.cvtColor(hsv_t_0, cv2.COLOR_HSV2BGR)
    PIL.Image.fromarray(bgr_t_0).save("{}/flow_t_0_t={}.jpg".format(save_path, args.t))

    flow_t_1 = flow_t_1.squeeze(0).numpy().transpose((1, 2, 0))
    # get rid of padding after prediction
    if width_pad > 0:
        flow_t_1 = flow_t_1[:,width_pad:,:]
    if height_pad > 0:
        flow_t_1 = flow_t_1[height_pad:,:,:]
    hsv_t_1 = np.zeros((flow_t_1.shape[0], flow_t_1.shape[1], 3), dtype=np.uint8)
    hsv_t_1[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow_t_1[..., 0], flow_t_1[..., 1])
    hsv_t_1[..., 0] = ang * 180 / np.pi / 2
    hsv_t_1[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr_t_1 = cv2.cvtColor(hsv_t_1, cv2.COLOR_HSV2BGR)
    PIL.Image.fromarray(bgr_t_1).save("{}/flow_t_1_t={}.jpg".format(save_path, args.t))

    # save bidirectional weight maps
    w1 = w1.squeeze().numpy()
    # get rid of padding after prediction
    if width_pad > 0:
        w1 = w1[:,width_pad:]
    if height_pad > 0:
        w1 = w1[height_pad:,:]
    PIL.Image.fromarray((w1 * 255).astype(np.uint8), 'L').save("{}/weight_map_t_0_t={}.jpg".format(save_path, args.t))
    w2 = w2.squeeze().numpy()
    # get rid of padding after prediction
    if width_pad > 0:
        w2 = w2[:,width_pad:]
    if height_pad > 0:
        w2 = w2[height_pad:,:]
    PIL.Image.fromarray((w2 * 255).astype(np.uint8), 'L').save("{}/weight_map_t_1_t={}.jpg".format(save_path, args.t))