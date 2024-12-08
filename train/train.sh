#!/bin/bash

# Set environment variables (if needed)
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0, adjust if needed

# Command to run the Python script with your arguments
python train.py \
  --path_dir_train "/home/alvin/UltrAi/Datasets/raw_datasets/m4raw/sample/multicoil_train" \
  --path_dir_test "/home/alvin/UltrAi/Datasets/raw_datasets/m4raw/sample/multicoil_val" \
  --acc 2 \
  --frac_c 0.16 \
  --img_size 256 \
  --bhsz 1 \
  --save_dir "../saved_models/m4raw_sample" \
  --exp_name "kspace_cold_diffusion_coilwise" \
  --save_every 200 \
  --num_epochs 50 \
  --learning_rate 2e-5 \
  --time_steps 1000