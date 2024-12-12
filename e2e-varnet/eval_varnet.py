import fastmri
import torch
import numpy as np
from fastmri.data.mri_data import SliceDataset
from fastmri.models import VarNet
from fastmri.data.subsample import MaskFunc, RandomMaskFunc,EquiSpacedMaskFunc,EquispacedMaskFractionFunc
from fastmri.data import transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.functional import structural_similarity_index_measure as SSIM
from torchmetrics.functional import peak_signal_noise_ratio as PSNR

# gaussian mask
class GaussianMaskFunc(MaskFunc):
    """
    Creates a Gaussian sub-sampling mask of a given shape.

    The mask selects k-space lines with probabilities following a Gaussian distribution.
    This prioritizes selecting lines near the center of k-space more frequently,
    with fewer lines sampled at the edges.
    """

    def __init__(
        self,
        center_fractions: Sequence[float],
        accelerations: Sequence[int],
        std_dev_factor: float = 0.15,
        allow_any_combination: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Args:
            center_fractions: Fraction of low-frequency columns to be retained.
            accelerations: Amount of under-sampling.
            std_dev_factor: Controls the spread of the Gaussian distribution relative to the k-space width.
                Smaller values lead to tighter central sampling.
            allow_any_combination: Allow any combination of center_fractions and accelerations.
            seed: Seed for reproducibility.
        """
        super().__init__(center_fractions, accelerations, allow_any_combination, seed)
        self.std_dev_factor = std_dev_factor

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        """
        Produce mask for non-central acceleration lines using a Gaussian distribution.

        Args:
            num_cols: Number of columns of k-space (2D subsampling).
            acceleration: Desired acceleration rate.
            offset: Not used for Gaussian sampling.
            num_low_frequencies: Number of low-frequency lines sampled (center).

        Returns:
            A mask for the high spatial frequencies of k-space.
        """
        # generate a gaussian probability distribution
        center = num_cols // 2
        std_dev = num_cols * self.std_dev_factor
        probabilities = np.exp(-((np.arange(num_cols) - center) ** 2) / (2 * std_dev**2))

        # normalize probabilities to ensure expected acceleration
        probabilities /= probabilities.sum()
        target_num_samples = num_cols / acceleration
        probabilities *= target_num_samples

        # sample based on the probabilities
        mask = self.rng.uniform(size=num_cols) < probabilities

        # ensure center frequencies are fully sampled
        mask[:num_low_frequencies] = 1
        mask[-num_low_frequencies:] = 1

        return mask
    
# fastmri dataset, change mask if need
dataset = SliceDataset(root='/home/lixin224/lixin-project/eece571f/M4Raw/multicoil_val_frac',challenge='multicoil',
                       transform=T.VarNetDataTransform(EquispacedMaskFractionFunc(center_fractions=[0.08], accelerations=[4])))
val_loader = DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=4,drop_last=False,pin_memory=True)

d = dataset[0]
for i in d:
    try:
        print(i.shape)
    except:
        print(i)

ssim_list = []
psnr_list = []

# build model load weight
model = VarNet(num_cascades=12,sens_chans=8,sens_pools=4,chans=18,pools=4).cuda()
checkpoint = torch.load('/home/lixin224/lixin-scratch/eece571f/varnet/varnet_demo/checkpoints/epoch=35-step=419256.ckpt')["state_dict"]

checkpoint = {k.replace("varnet.", "",1): v for k, v in checkpoint.items()}
del checkpoint["loss.w"]
model.load_state_dict(checkpoint, strict=True)
del checkpoint
model.eval()

# inference
for index,d in enumerate(tqdm(val_loader)):
    with torch.no_grad():
        pre = model(d[0].cuda(),d[1].cuda(),num_low_frequencies=d[2].cuda())
        ssim_list.append(SSIM(pre.unsqueeze(0),d[3].unsqueeze(0).cuda(),data_range = d[-2].cuda()).item())## 
        psnr_list.append(PSNR(pre.unsqueeze(0),d[3].unsqueeze(0).cuda(),data_range = d[-2].cuda()).item())
        
# SSIM and PSNR calculations
ssim_avg = round(sum(ssim_list) / len(ssim_list), 2)
psnr_avg = round(sum(psnr_list) / len(psnr_list), 2)

# write the results to results.txt
with open('results.txt', 'w') as f:
    f.write(f'SSIM: {ssim_avg}\nPSNR: {psnr_avg}\n')

