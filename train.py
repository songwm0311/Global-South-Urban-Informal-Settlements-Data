"""Train an urban informal-settlement segmentation model."""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import tqdm

from config_all import build_loaders
from loss import LossSelector
from model_factory import MODEL_NAMES, build_model, primary_output
from utils import AverageMeter


LOSS_NAMES = tuple(LossSelector)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train an informal-settlement segmentation model"
    )
    parser.add_argument("--data-dir", default="jm", help="Directory with x_train/ and y_train/")
    parser.add_argument("--out", default="result", help="Output directory")
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--lr", "--learning-rate", default=0.0001, type=float)
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--train-count", default=1000, type=int)
    parser.add_argument("--val-count", default=200, type=int)
    parser.add_argument("--test-count", default=200, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--in-channels", default=13, type=int)
    parser.add_argument("--model", default="Unet3Plus", choices=MODEL_NAMES)
    parser.add_argument("--loss", default="mseloss", choices=LOSS_NAMES)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or a specific CUDA device such as cuda:0",
    )
    return parser


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run_epoch(loader, model, optimizer, criterion, device, training, phase):
    losses = AverageMeter()
    errors = AverageMeter()
    rmses = AverageMeter()
    model.train(training)

    for inputs, labels, mask in tqdm(loader, desc=phase):
        inputs = inputs.float().to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)
        mask = mask.float().to(device, non_blocking=True)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            outputs = primary_output(model(inputs))
            loss = criterion(outputs, labels, mask)
            if training:
                loss.backward()
                optimizer.step()

        valid_pixels = mask.sum().clamp_min(1.0)
        absolute_error = torch.abs(outputs - labels) * mask
        mean_error = absolute_error.sum() / valid_pixels
        rmse = torch.sqrt((absolute_error**2).sum() / valid_pixels)

        batch_size = inputs.size(0)
        losses.update(loss.item(), batch_size)
        errors.update(mean_error.item(), batch_size)
        rmses.update(rmse.item(), batch_size)

    print(
        "{} — loss: {:.4f} | error: {:.4f} | RMSE: {:.4f}".format(
            phase, losses.avg, errors.avg, rmses.avg
        )
    )
    return losses.avg, errors.avg, rmses.avg


def unwrap_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def main(argv=None):
    args = build_parser().parse_args(argv)
    setup_seed(args.seed)
    device = resolve_device(args.device)

    train_loader, val_loader, test_loader = build_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_count=args.train_count,
        val_count=args.val_count,
        test_count=args.test_count,
    )

    model = build_model(args.model, in_channels=args.in_channels).to(device)
    if (
        device.type == "cuda"
        and device.index in (None, 0)
        and torch.cuda.device_count() > 1
    ):
        model = torch.nn.DataParallel(model)

    criterion = LossSelector[args.loss](weight=None)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    output_dir = Path(args.out)
    checkpoint_dir = output_dir / "checkpoints" / args.model
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(output_dir / "tensorboard"))
    best_validation_error = float("inf")

    print("device:", device)
    print("model:", args.model)
    print("loss:", args.loss)
    print("total parameters: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1e6))

    for epoch in range(args.epochs):
        started = time.time()
        print(
            "\nEpoch: [{} | {}] LR: {:.6f}".format(
                epoch + 1, args.epochs, optimizer.param_groups[0]["lr"]
            )
        )

        train_metrics = run_epoch(
            train_loader, model, optimizer, criterion, device, True, "TRAIN"
        )
        val_metrics = run_epoch(
            val_loader, model, optimizer, criterion, device, False, "VAL"
        )
        test_metrics = run_epoch(
            test_loader, model, optimizer, criterion, device, False, "TEST"
        )

        if val_metrics[1] < best_validation_error:
            best_validation_error = val_metrics[1]
            torch.save(
                unwrap_model(model).state_dict(), checkpoint_dir / "best_model.pt"
            )

        writer.add_scalars(
            "loss",
            {"train": train_metrics[0], "val": val_metrics[0], "test": test_metrics[0]},
            epoch,
        )
        writer.add_scalars(
            "mean_absolute_error",
            {"train": train_metrics[1], "val": val_metrics[1], "test": test_metrics[1]},
            epoch,
        )
        writer.add_scalars(
            "rmse",
            {"train": train_metrics[2], "val": val_metrics[2], "test": test_metrics[2]},
            epoch,
        )

        elapsed = time.time() - started
        print("epoch completed in {:.0f}m {:.0f}s".format(elapsed // 60, elapsed % 60))

    writer.close()


if __name__ == "__main__":
    main()
