"""Display a raster band or derived output without embedding study data."""
import argparse
import matplotlib.pyplot as plt
import rasterio

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("raster"); p.add_argument("--band", type=int, default=1)
    p.add_argument("--output"); a = p.parse_args()
    with rasterio.open(a.raster) as src: array = src.read(a.band, masked=True)
    plt.figure(figsize=(8, 6)); plt.imshow(array); plt.colorbar(); plt.axis("off"); plt.tight_layout()
    if a.output: plt.savefig(a.output, dpi=300, bbox_inches="tight")
    else: plt.show()

