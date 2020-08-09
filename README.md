# Video Frame Interpolation

## Docker
`nvidia-docker run --rm -v ~/samson:/workspace/samson -it video-interpolate:latest`

## TODO:
- [X] Make the convolutional autoencoder
- [X] Add validation split
- [X] Dense optical flow
- [X] PSNR evaluation metric for SOTA comparisons
- [X] Skip connections
- [X] Argparse
- [ ] Different image sizes?
- [X] Saving statistics
- [X] Plotting statistics
- [X] Video maker