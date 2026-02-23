import torch
from torch.utils.data import Dataset
import cv2
import os
import numpy as np

from models.classical.dark_channel import get_dark_channel
from models.classical.atmospheric_light import estimate_atmospheric_light
from models.classical.transmission import estimate_transmission
from models.classical.guided_filter import guided_filter


class ResidualTransmissionDataset(Dataset):

    def __init__(self, root_dir):
        self.hazy_dir = os.path.join(root_dir, "hazy")
        self.trans_dir = os.path.join(root_dir, "transmission")

        self.files = os.listdir(self.hazy_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        filename = self.files[idx]

        hazy = cv2.imread(os.path.join(self.hazy_dir, filename))
        hazy = cv2.cvtColor(hazy, cv2.COLOR_BGR2RGB)
        hazy = hazy.astype(np.float32) / 255.0

        gt_trans = cv2.imread(os.path.join(self.trans_dir, filename), 0)
        gt_trans = gt_trans.astype(np.float32) / 255.0

        # Classical transmission estimation
        dark = get_dark_channel(hazy)
        A = estimate_atmospheric_light(hazy, dark)
        raw_trans = estimate_transmission(hazy, A)

        gray = cv2.cvtColor((hazy * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray = gray.astype(np.float32) / 255.0
        classical_trans = guided_filter(gray, raw_trans)

        # Compute residual
        residual = gt_trans - classical_trans

        hazy_tensor = torch.tensor(hazy).permute(2, 0, 1).float()
        classical_tensor = torch.tensor(classical_trans).unsqueeze(0).float()
        residual_tensor = torch.tensor(residual).unsqueeze(0).float()

        # Input now has 4 channels: RGB + Classical Transmission
        input_tensor = torch.cat([hazy_tensor, classical_tensor], dim=0)

        return input_tensor, residual_tensor