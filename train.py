"""Train and evaluate one manifest-selected outer fold."""
import argparse
from pathlib import Path
import config_all as cfg
from nameList import load_city_files
from workflow import run_outer_fold

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--test-city", required=True)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--epochs", type=int, default=cfg.FINAL_TRAINING_EPOCHS)
    args = parser.parse_args()
    files = load_city_files(args.manifest, args.data_root)
    print(run_outer_fold(files, args.test_city, Path(args.output_dir).resolve(), args.epochs))

