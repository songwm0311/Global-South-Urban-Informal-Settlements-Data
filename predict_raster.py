"""Run tiled inference for one multi-band MATLAB raster."""

import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from tqdm import tqdm

from dataloder_Pick import load_mat_array
from load_model import load_checkpoint, resolve_device
from model_factory import MODEL_NAMES, build_model, primary_output


def tile_positions(length, tile_size, stride):
    if length <= tile_size:
        return [0]
    positions = list(range(0, length - tile_size + 1, stride))
    final_position = length - tile_size
    if positions[-1] != final_position:
        positions.append(final_position)
    return positions


def pad_to_tile(data, tile_size):
    height, width, _ = data.shape
    pad_height = max(0, tile_size - height)
    pad_width = max(0, tile_size - width)
    if not pad_height and not pad_width:
        return data
    mode = "reflect" if height > 1 and width > 1 else "edge"
    return np.pad(data, ((0, pad_height), (0, pad_width), (0, 0)), mode=mode)


def predict_raster(model, data, device, tile_size=128, overlap=40):
    """Predict a probability mosaic by averaging overlapping tiles."""
    if not 0 <= overlap < tile_size:
        raise ValueError("overlap must be at least 0 and smaller than tile_size")
    original_height, original_width = data.shape[:2]
    data = pad_to_tile(data, tile_size)
    height, width = data.shape[:2]
    stride = tile_size - overlap
    rows = tile_positions(height, tile_size, stride)
    columns = tile_positions(width, tile_size, stride)

    probability_sum = np.zeros((height, width), dtype=np.float32)
    observation_count = np.zeros((height, width), dtype=np.float32)
    model.eval()

    with torch.no_grad():
        for row in tqdm(rows, desc="PREDICT"):
            for column in columns:
                patch = data[row : row + tile_size, column : column + tile_size]
                tensor = torch.from_numpy(np.transpose(patch, (2, 0, 1))).unsqueeze(0)
                tensor = tensor.float().to(device)
                probability = primary_output(model(tensor))[0, 0].cpu().numpy()
                probability_sum[
                    row : row + tile_size, column : column + tile_size
                ] += probability
                observation_count[
                    row : row + tile_size, column : column + tile_size
                ] += 1

    probability = probability_sum / np.maximum(observation_count, 1)
    return probability[:original_height, :original_width]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Predict informal settlements in a multi-band MATLAB raster"
    )
    parser.add_argument("--input", required=True, help="Input .mat raster (H x W x C)")
    parser.add_argument("--checkpoint", required=True, help="Model state dictionary")
    parser.add_argument("--output", required=True, help="Output .mat file")
    parser.add_argument("--model", default="Unet3Plus", choices=MODEL_NAMES)
    parser.add_argument("--in-channels", default=13, type=int)
    parser.add_argument("--tile-size", default=128, type=int)
    parser.add_argument("--overlap", default=40, type=int)
    parser.add_argument("--threshold", default=0.65, type=float)
    parser.add_argument("--scale-min", default=500.0, type=float)
    parser.add_argument("--scale-max", default=9000.0, type=float)
    parser.add_argument(
        "--input-normalized",
        action="store_true",
        help="Skip reflectance scaling when the input is already in [0, 1]",
    )
    parser.add_argument("--device", default="auto")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    data = np.squeeze(load_mat_array(args.input)).astype(np.float32, copy=False)
    if data.ndim != 3 or data.shape[2] != args.in_channels:
        raise ValueError(
            "Input must have shape H x W x {}; found {}".format(
                args.in_channels, data.shape
            )
        )
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    if args.input_normalized:
        data = np.clip(data, 0.0, 1.0)
    else:
        if args.scale_max <= args.scale_min:
            raise ValueError("scale-max must be greater than scale-min")
        data = np.clip(
            (data - args.scale_min) / (args.scale_max - args.scale_min), 0.0, 1.0
        )

    device = resolve_device(args.device)
    model = build_model(args.model, in_channels=args.in_channels).to(device)
    load_checkpoint(model, args.checkpoint, device)
    probability = predict_raster(
        model, data, device, tile_size=args.tile_size, overlap=args.overlap
    )
    prediction = (probability >= args.threshold).astype(np.uint8)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(output, {"prediction": prediction, "probability": probability})


if __name__ == "__main__":
    main()
