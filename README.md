# Video Frame Interpolation

## Requirements

## Features
- [X] Create dataloader: `./dataloader.py`
- [X] Add validation split
- [X] Add discriminator to state-of-the-art model RRIN and mark changes: `./model.py`
- [X] Argparse for custom settings
- [X] Evaluation metrics (peak signal-to-noise ratio and structural similarity index measure): `./evaluate.py`
- [X] Training/validation loops and saving statistics: `./train.py`
- [X] Plotting statistics: `./plot_stats.py`
- [X] Video converter: `./convert_vid.py`
- [X] Visualisations: optical flow estimates and weight maps

## Visualisations for GUI
- Only multiples of 8

## References
- Vimeo-90k: http://toflow.csail.mit.edu/
- U-Net: https://github.com/jvanvugt/pytorch-unet
- RRIN
- PSNR and SIIM: https://scikit-image.org/docs/dev/api/skimage.metrics.html