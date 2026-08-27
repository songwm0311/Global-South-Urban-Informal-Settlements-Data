"""Smoke-test a bundled segmentation architecture with a synthetic input."""

import argparse

import torch

from model_factory import MODEL_NAMES, build_model, primary_output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Unet3Plus", choices=MODEL_NAMES)
    parser.add_argument("--in-channels", default=13, type=int)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--height", default=128, type=int)
    parser.add_argument("--width", default=128, type=int)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    model = build_model(args.model, in_channels=args.in_channels).to(device)
    inputs = torch.randn(
        args.batch_size,
        args.in_channels,
        args.height,
        args.width,
        device=device,
    )
    with torch.no_grad():
        output = primary_output(model(inputs))
    print("input:", tuple(inputs.shape))
    print("output:", tuple(output.shape))
    print("parameters:", sum(parameter.numel() for parameter in model.parameters()))


if __name__ == "__main__":
    main()
