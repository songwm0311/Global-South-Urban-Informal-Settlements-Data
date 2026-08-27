"""Load a trained segmentation model and run inference on patch datasets."""

import argparse
from pathlib import Path

import scipy.io as sio
import torch
from tqdm import tqdm

from config_all import build_loaders
from model_factory import MODEL_NAMES, build_model, primary_output


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def load_checkpoint(model, checkpoint, device):
    """Load plain or wrapped state dictionaries, including DataParallel keys."""
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError("Checkpoint must contain a PyTorch state dictionary")
    state = {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state.items()
    }
    model.load_state_dict(state)
    return model


def predict_loader(model, loader, device, output_dir, threshold=0.65):
    """Save prediction, label, and validity mask for every evaluation batch."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    with torch.no_grad():
        for batch_index, (inputs, labels, mask) in enumerate(
            tqdm(loader, desc="PREDICT"), start=1
        ):
            inputs = inputs.float().to(device, non_blocking=True)
            probabilities = primary_output(model(inputs))
            predictions = (probabilities >= threshold).to(torch.uint8)
            sio.savemat(
                output_dir / "batch-{:04d}.mat".format(batch_index),
                {
                    "prediction": predictions.cpu().numpy(),
                    "probability": probabilities.cpu().numpy(),
                    "label": labels.numpy(),
                    "mask": mask.numpy(),
                },
            )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run patch-based informal-settlement inference"
    )
    parser.add_argument("--checkpoint", required=True, help="Path to model weights")
    parser.add_argument("--data-dir", default="jm", help="Directory with x_train/ and y_train/")
    parser.add_argument("--output-dir", default="predictions")
    parser.add_argument("--model", default="Unet3Plus", choices=MODEL_NAMES)
    parser.add_argument("--in-channels", default=13, type=int)
    parser.add_argument("--threshold", default=0.65, type=float)
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--train-count", default=1000, type=int)
    parser.add_argument("--val-count", default=200, type=int)
    parser.add_argument("--test-count", default=200, type=int)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    _, _, test_loader = build_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_count=args.train_count,
        val_count=args.val_count,
        test_count=args.test_count,
    )
    model = build_model(args.model, in_channels=args.in_channels).to(device)
    load_checkpoint(model, args.checkpoint, device)
    predict_loader(model, test_loader, device, args.output_dir, args.threshold)


if __name__ == "__main__":
    main()
