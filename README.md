# Video Frame Interpolation
## Requirements
torch, numpy, cv2, argparse, PIL, skimage.metrics, matplotlib

## Features
- [X] Create dataloader: `dataloader.py`
- [X] Add validation split
- [X] Add discriminator to state-of-the-art model RRIN and mark changes: `model.py`
- [X] Argparse for custom settings
- [X] Evaluation metrics (PSNR and SSIM): `evaluate.py`
- [X] Training/validation loops and saving statistics: `train.py`
- [X] Plotting statistics: `plot_stats.py`
- [X] Generate interpolated frame, optical flow estimates and weight maps: `generate.py`
- [X] Video converter: `convert_vid.py`

## Results
### Pretained Weights
The pretrained weights for our 4 experiments can be found in the `weights` folder, with the format `model_{learning_rate}_{batch_size}.pt`.

### Experiment Statistics
The experimental data folders for our 4 experiments can be found in the `exps` folder, with the format `exps_{learning_rate}_{batch_size}`. Refer to the "Plot Statistics" section to know how to visualise the data.

### Generated Images
Some of our generated samples can be found in the `results` folder:
- `happiness_facial`: results for a facial expression of happiness from the Human ID Project at the The University of Texas at Dallas.
- `vimeo_90k_test`: results for the triplet `00001/0830` in Vimeo-90k, which is part of the test set.

## Running the Code
### Training
`python3 train.py --vimeo_90k_path /path/to/vimeo-90k/ --save_stats_path /path/to/folder/to/save/experiment/details/ --save_model_path /path/to/save/model/weights.pt`

- This trains a new model on the Vimeo-90k train set.
- You can specify `--num_epochs`, `--batch_size` and `--lr` as hyperparameters.
- Use `--eval_every` to specify how often we evaluate the model using the validation set, and save the losses.
- Use `--max_num_images` if you do not want to train on the whole dataset.
- Specify `--timeit` if you want timing estimates.
- Use `--time_check_every` to decide how often you want timing estimates, based on the number of batches per interval.

### Plot Statistics
`python3 plot_stats.py --exp_dir path/to/experiment/folder/`

- This plots the loss graphs for an experiment

### Model Evaluation
`python3 evaluate.py --vimeo_90k_path /path/to/vimeo-90k/ --saved_model_path /path/to/model/weights.pt`

- This evaluates a model on the Vimeo-90k test set for PSNR and SSIM.

### Generation
`python3 generate.py --frames_path /path/to/folder/with/target/frames/ --saved_model_path /path/to/model/weights.pt`

- This generates the interpolated middle frame of 2 frames, and the corresponding optical flow estimates and weight maps.
- The frames in your `--frames_path` will be sorted, only the first and second frames will be used.
- A folder containing the outputs will be created in your `--frames_path`.
- This is optional, but you can set the timestep for interpolation with `--t`. It can range from 0 to 1, with 0.5 being the midpoint and the default value.
- NOTE: The two input frames must have the same size, but can handle inputs of variable sizes otherwise.

### Video Conversion
`python3 convert_vid.py --vid_path /path/to/input/video.mp4 --save_vid_path /path/to/save/video.mp4 --saved_model_path /path/to/model/weights.pt`

- You can use `--print_every` to specify the frame interval for printing progress.
- NOTE: The two input frames must have the same size, but can handle inputs of variable sizes otherwise.

## References:
- https://github.com/HopLee6/RRIN