import sys
sys.path.append('../')
import argparse
import pathlib
import torch
import numpy as np
import os
from torch.utils.data import DataLoader

from utils.mri_data import SliceDataset
from utils.data_transform import DataTransform_Diffusion
from utils.sample_mask import RandomMaskDiffusion
from utils.misc import create_path, calc_model_size
from diffusion.kspace_diffusion import KspaceDiffusion
from utils.diffusion_train import Trainer
from net.u_net_diffusion import Unet

from glob import glob
from natsort import natsorted


# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Training script for diffusion model.")
    
    # Add arguments for training settings
    parser.add_argument('--acc', type=int, default=2, help='Acceleration factor')
    parser.add_argument('--frac_c', type=float, default=0.16, help='Center fraction')
    parser.add_argument('--path_dir_train', type=str, required=True, help='Path to training dataset')
    parser.add_argument('--path_dir_test', type=str, required=True, help='Path to test dataset')
    parser.add_argument('--img_size', type=int, default=256, help='Image size')
    parser.add_argument('--bhsz', type=int, default=1, help='Batch size')
    parser.add_argument('--save_dir', type=str, default="../saved_models", help='Directory to save the models')
    parser.add_argument('--exp_name', type=str, default="proposed_multicoil", help='Experiment name')
    parser.add_argument('--multicoil', action='store_true', help='Use multicoil data')
    parser.add_argument('--kspace_diff', action='store_true', help='Use k-space diffusion')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--time_steps', type=int, default=1000, help='Number of diffusion time steps')
    parser.add_argument('--save_every', type=int, default=15000, help='Save model every n steps')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--test_steps', type=int, default=125, help='Number of test steps')
    parser.add_argument('--load_path', type=str, default=None, help='Path to load model.')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    for arg in vars(args):
        print(arg, getattr(args, arg))

    # Setup environment
    print(torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    # Dataset settings
    acc = args.acc
    frac_c = args.frac_c
    path_dir_train = args.path_dir_train
    path_dir_test = args.path_dir_test
    img_size = args.img_size
    bhsz = args.bhsz
    save_dir = args.save_dir
    exp_name = args.exp_name

    # Initialize mask
    mask_func = RandomMaskDiffusion(
        acceleration=acc,
        center_fraction=frac_c,
        size=(1, img_size, img_size),
    )

    # Initialize dataset
    data_transform = DataTransform_Diffusion(
        mask_func,
        img_size=img_size,
        combine_coil=True,
        flag_singlecoil=False,
    )

    # Training set
    dataset_train = SliceDataset(
        root=pathlib.Path(path_dir_train),
        transform=data_transform,
        challenge='multicoil',
        num_skip_slice=5,
    )

    # Test set
    dataset_test = SliceDataset(
        root=pathlib.Path(path_dir_test),
        transform=data_transform,
        challenge='multicoil',
        num_skip_slice=5,
    )

    dataloader_train = DataLoader(dataset_train, batch_size=bhsz, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=bhsz, shuffle=True)
    print('len dataloader train:', len(dataloader_train))
    print('len dataloader test:', len(dataloader_test))

    # Model settings
    CH_MID = 64
    NUM_EPOCH = args.num_epochs
    learning_rate = args.learning_rate
    time_steps = args.time_steps
    train_steps = NUM_EPOCH * len(dataloader_train)
    # train_steps = max(train_steps, 700000)
    train_steps = train_steps if args.multicoil else train_steps // 4 # each coil counts as one iteration for single coil approach
    print('train_steps:', train_steps)

    # Save settings
    PATH_MODEL = f'{save_dir}/{exp_name}_{acc}x_T{time_steps}_S{train_steps}/'
    create_path(PATH_MODEL)

    # Construct diffusion model
    channels=8 if args.multicoil else 2
    model = Unet(
        dim=CH_MID,
        dim_mults=(1, 2, 4, 8),
        channels=channels
    ).cuda()
    print('model size: %.3f MB' % (calc_model_size(model)))

    diffusion = KspaceDiffusion(
        model,
        image_size=img_size,
        device_of_kernel=device,
        channels=channels,
        timesteps=time_steps,
        loss_type='l1',
        blur_routine='Constant',
        train_routine='Final',
        sampling_routine='x0_step_down',
        discrete=False,
        multicoil=args.multicoil,
        kspace_diff=args.kspace_diff
    ).cuda()

    # obtain load path
    if args.load_path is None:
        ckpts = glob(f'{PATH_MODEL}/model_*.pt')
        args.load_path = natsorted(ckpts)[-1] if ckpts else None

    # Construct trainer and train
    trainer = Trainer(
        diffusion,
        image_size=img_size,
        train_batch_size=bhsz,
        train_lr=learning_rate,
        train_num_steps=train_steps,
        gradient_accumulate_every=2,
        ema_decay=0.995,
        fp16=False,
        save_and_sample_every=args.save_every if args.multicoil else args.save_every // 4,
        results_folder=PATH_MODEL,
        load_path=args.load_path,
        dataloader_train=dataloader_train,
        dataloader_test=dataloader_test,
    )

    if args.train:
        print(f"\nTraining with: kspace_diff={args.kspace_diff}, multicoil={args.multicoil}")
        trainer.train()
    else:
        print("Testing...")
        trainer.test(t=args.test_steps)

if __name__ == "__main__":
    main()
