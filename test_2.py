"""Run the complete manifest-defined leave-one-city-out evaluation."""
import argparse
from pathlib import Path
import config_all as cfg
from nameList import load_city_files
from workflow import run_all

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--epochs", type=int, default=cfg.FINAL_TRAINING_EPOCHS)
    args = parser.parse_args()
    print(run_all(load_city_files(args.manifest, args.data_root), Path(args.output_dir).resolve(), args.epochs))

