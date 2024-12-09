#!/bin/bash

### GPU batch job ###
#SBATCH --job-name=kspace_cold_diffusion
#SBATCH --account=st-ilker-1-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=2
#SBATCH --mem=64G
#SBATCH --constraint=gpu_mem_32
#SBATCH --time=3-10:00:00
#SBATCH --output=outputs/%x-%j_output.txt
#SBATCH --error=outputs/%x-%j_error.txt
# #SBATCH --mail-user=alvinbk@student.ubc.ca
# #SBATCH --mail-type=ALL

#############################################################################

source ~/.bashrc
conda activate kspace_cold

# Command to run the Python script with your arguments
python train.py \
  --path_dir_train "/home/alvinbk/project/EECE571/datasets/m4raw/multicoil_train" \
  --path_dir_test "/home/alvinbk/project/EECE571/datasets/m4raw/multicoil_val" \
  --acc 2 \
  --frac_c 0.16 \
  --img_size 256 \
  --bhsz 1 \
  --save_dir "../saved_models" \
  --exp_name "kspace_cold_diffusion_multicoil" \
  --save_every 15000 \
  --num_epochs 50 \
  --learning_rate 2e-5 \
  --time_steps 1000 \
  --multicoil \
  --train