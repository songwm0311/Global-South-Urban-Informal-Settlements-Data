"""Load a saved U-Net checkpoint for inference."""
import torch
import config_all as cfg
from model import UNet


def load_model(checkpoint_path, device=cfg.DEVICE):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    active = checkpoint["active_band_names"]
    model = UNet(len(active) + 5).to(device)
    model.load_state_dict(checkpoint["state_dict"]); model.eval()
    return model, active, checkpoint

