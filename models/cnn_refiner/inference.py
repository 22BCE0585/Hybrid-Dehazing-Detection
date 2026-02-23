import torch
import cv2
import numpy as np

from models.cnn_refiner.model import TransmissionRefiner
from models.classical.dark_channel import get_dark_channel
from models.classical.atmospheric_light import estimate_atmospheric_light
from models.classical.transmission import estimate_transmission
from models.classical.guided_filter import guided_filter


class CNNRefiner:

    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = TransmissionRefiner().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def refine(self, image):

        # Classical transmission
        dark = get_dark_channel(image)
        A = estimate_atmospheric_light(image, dark)
        raw_trans = estimate_transmission(image, A)

        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray = gray.astype(np.float32) / 255.0
        classical_trans = guided_filter(gray, raw_trans)

        # Prepare 4-channel input
        hazy_tensor = torch.tensor(image).permute(2, 0, 1).float()
        classical_tensor = torch.tensor(classical_trans).unsqueeze(0).float()

        input_tensor = torch.cat([hazy_tensor, classical_tensor], dim=0)
        input_tensor = input_tensor.unsqueeze(0).to(self.device)

        # Predict residual
        with torch.no_grad():
            residual = self.model(input_tensor)

        residual = residual.squeeze().cpu().numpy()

        # Final transmission
        # Resize residual to match classical transmission if needed
        if residual.shape != classical_trans.shape:
            residual = cv2.resize(
                residual,
                (classical_trans.shape[1], classical_trans.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

        refined_trans = classical_trans + residual
        refined_trans = np.clip(refined_trans, 0, 1)

        return refined_trans, A