"""Generate a probability GeoTIFF with one saved checkpoint."""
import argparse
from pathlib import Path
import numpy as np
from evaluation import write_raster
from load_model import load_model
from nameList import load_city_files
from preprocessing import CityScene
from training import predict_full_scene

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--manifest", required=True); p.add_argument("--data-root")
    p.add_argument("--city", required=True); p.add_argument("--checkpoint", required=True); p.add_argument("--output", required=True)
    a = p.parse_args(); files = load_city_files(a.manifest, a.data_root); paths = files[a.city]
    scene = CityScene(a.city, paths["image"], paths["shp"]); model, active, _ = load_model(a.checkpoint)
    probability = predict_full_scene(model, scene, active)
    write_raster(Path(a.output), np.where(scene.valid, probability, -9999.), scene, "float32", -9999.)

