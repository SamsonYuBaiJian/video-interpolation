# Video Frame Interpolation

## Requirements

## Docker
`nvidia-docker run --rm -v ~/samson:/workspace/samson -it video-interpolate:latest`

## Features
- [X] Create dataloader
- [X] Add validation split
- [X] Add adversarial training to RRIN
- [X] Argparse
- [X] Evaluation metrics (PSNR and SSIM)
- [X] Saving statistics
- [X] Plotting statistics
- [X] Add classic interpolation methods
- [X] Video converter
- [X] Optical flow and weight map visualisations

## Dataloader
- The dataloader for the Vimeo-90k dataset can be found in `./utils.py`.
- The data processing we have done includes:
  - ABC

## Model

## Evaluation Metrics
- PSNR
- SSIM

## Statistics

## Visualisations for GUI
- Only multiples of 64

## Video Converter

## References
- Vimeo-90k
- UNet: https://github.com/jvanvugt/pytorch-unet
- RRIN