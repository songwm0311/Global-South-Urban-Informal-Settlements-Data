"""Plot city-level segmentation metrics produced by test_2.py."""
import argparse
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("metrics_csv"); p.add_argument("--output", default="LOCO_metrics.png")
    a = p.parse_args(); frame = pd.read_csv(a.metrics_csv, dtype={"Test_city": str})
    ax = frame.set_index("Test_city")[["Precision", "Recall", "F1-Score"]].plot.bar(figsize=(12, 6))
    ax.set(ylim=(0, 1), xlabel="Independent test city", ylabel="Score")
    ax.grid(axis="y", alpha=.25); ax.legend(frameon=False); plt.tight_layout(); plt.savefig(a.output, dpi=300)

