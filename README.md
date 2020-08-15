# Video Frame Interpolation
## Requirements
- torch
- numpy
- cv2
- argparse
- PIL
- skimage.metrics
- matplotlib

## Features
- [X] Create dataloader: `./dataloader.py`
- [X] Add validation split
- [X] Add discriminator to state-of-the-art model RRIN and mark changes: `./model.py`
- [X] Argparse for custom settings
- [X] Evaluation metrics (peak signal-to-noise ratio and structural similarity index measure): `./evaluate.py`
- [X] Training/validation loops and saving statistics: `./train.py`
- [X] Plotting statistics: `./plot_stats.py`
- [X] Generate interpolated frame, optical flow estimates and weight maps: `./generate.py`
- [X] Video converter: `./convert_vid.py`

## Training

## Generation
This generates the interpolated middle frame of 2 frames, and the corresponding optical flow estimates and weight maps.

`python3 generate.py --frames_path /path/to/folder/with/two/frames/ --saved_model_path /path/to/model/weights.pt --t 0.5`

## Video Conversion

- Visualisations: `python3 visualise.py --frames_path /path/to/folder/with/images --saved_model_path /path/to/model/weights.pt`

## References:
- https://github.com/HopLee6/RRIN