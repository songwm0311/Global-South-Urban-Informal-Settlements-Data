"""Region-neutral nested OOF training and independent evaluation workflow."""
import gc
import json
import numpy as np
import pandas as pd
import torch
import config as cfg
from evaluation import apply_operating_point, calculate_metrics, select_operating_point, write_raster
from preprocessing import load_scenes, make_indicator_maps
from training import predict_full_scene, train_model


def configure_output(output_dir):
    cfg.OUTPUT_DIR = output_dir
    cfg.MODEL_DIR = output_dir / "models"
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg.MODEL_DIR.mkdir(parents=True, exist_ok=True)


def run_outer_fold(city_files, test_city, output_dir, epochs=cfg.FINAL_TRAINING_EPOCHS):
    """Run one independent outer city; city identifiers come only from manifest."""
    configure_output(output_dir)
    if test_city not in city_files: raise ValueError(f"Unknown city_id: {test_city}")
    if len(city_files) < 3: raise ValueError("Nested OOF requires at least three cities")
    scenes, active = load_scenes(city_files, cfg.OUTPUT_DIR)
    development = [city for city in city_files if city != test_city]
    validation_items, test_members, histories, paths = [], [], [], []
    for validation_city in development:
        training_cities = [city for city in development if city != validation_city]
        tag = f"outer_{test_city}_inner_val_{validation_city}"
        model, history, path = train_model(test_city, training_cities,
                                           f"inner_validation_{validation_city}", epochs,
                                           scenes, active, tag)
        histories.extend(history); paths.append(str(path))
        validation_items.append((validation_city, predict_full_scene(model, scenes[validation_city], active), scenes[validation_city]))
        test_members.append(predict_full_scene(model, scenes[test_city], active))
        del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    threshold, max_fraction, min_pixels, validation_scores, detail = select_operating_point(validation_items)
    stack = np.stack(test_members).astype(np.float32)
    probability, probability_std = stack.mean(0), stack.std(0)
    scene = scenes[test_city]
    raw = (probability >= threshold).astype(np.uint8)
    prediction, effective = apply_operating_point(probability, scene.valid, threshold, max_fraction, min_pixels)
    row = calculate_metrics(test_city, probability, prediction, scene, development,
                            threshold, effective, max_fraction, min_pixels)
    row.update({"Validation_cities": ",".join(development), "Ensemble_members": len(test_members),
                "Model_paths": ";".join(paths), **validation_scores})
    pd.DataFrame(histories).to_csv(output_dir / f"{test_city}_training_history.csv", index=False)
    pd.DataFrame([{"Outer_test_city": test_city, **item} for item in detail]).to_csv(
        output_dir / f"{test_city}_OOF_validation_details.csv", index=False)
    city_dir = output_dir / f"test_{test_city}"; city_dir.mkdir(exist_ok=True)
    uisi, glcm_mean, glcm_variance = make_indicator_maps(scene)
    outputs = (
        ("UISI", uisi, "float32", -9999.), ("GLCM_mean", glcm_mean, "float32", -9999.),
        ("GLCM_variance", glcm_variance, "float32", -9999.),
        ("ensemble_probability", probability, "float32", -9999.),
        ("ensemble_probability_std", probability_std, "float32", -9999.),
        ("prediction_raw_binary", raw, "uint8", 255), ("prediction_binary", prediction, "uint8", 255),
        ("ground_truth", scene.truth, "uint8", 255))
    for name, array, dtype, nodata in outputs:
        fill = np.where(scene.valid, array, nodata)
        write_raster(city_dir / f"{test_city}_{name}.tif", fill, scene, dtype, nodata)
    metadata = {"test_city": test_city, "development_cities": development, "epochs_per_member": epochs,
                "random_seed": cfg.RANDOM_SEED, "device": str(cfg.DEVICE), "active_band_names": active,
                "derived_channels": ["NDVI", "local_NIR_std", "UISI", "GLCM_mean", "GLCM_variance"],
                "glcm_levels": cfg.GLCM_LEVELS, "glcm_distance": cfg.GLCM_DISTANCE,
                "glcm_direction": list(cfg.GLCM_DIRECTION)}
    (output_dir / f"{test_city}_run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return row


def run_all(city_files, output_dir, epochs=cfg.FINAL_TRAINING_EPOCHS):
    rows = [run_outer_fold(city_files, city, output_dir, epochs) for city in city_files]
    frame = pd.DataFrame(rows); frame.to_csv(output_dir / "all_city_LOCO_metrics.csv", index=False)
    return frame

