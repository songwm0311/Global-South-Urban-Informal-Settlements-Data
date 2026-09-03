"""OOF operating-point selection, metrics, and GeoTIFF output."""
import numpy as np
import rasterio
from scipy import ndimage
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
import config as cfg


def remove_small_components(binary, valid, minimum):
    binary = (binary.astype(bool) & valid).astype(np.uint8)
    if minimum <= 1 or not binary.sum(): return binary
    labels, count = ndimage.label(binary, structure=np.ones((3, 3), dtype=np.uint8))
    if not count: return binary
    keep = np.bincount(labels.ravel()) >= int(minimum); keep[0] = False
    return keep[labels].astype(np.uint8)


def confusion_metrics(truth, prediction, valid, beta=cfg.FBETA_BETA):
    y, p = truth[valid].astype(np.uint8), prediction[valid].astype(np.uint8)
    tp, fp, fn = int(((y == 1) & (p == 1)).sum()), int(((y == 0) & (p == 1)).sum()), int(((y == 1) & (p == 0)).sum())
    precision, recall = tp / (tp + fp) if tp + fp else 0., tp / (tp + fn) if tp + fn else 0.
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
    b2 = beta ** 2; fbeta = (1 + b2) * precision * recall / (b2 * precision + recall) if precision + recall else 0.
    return precision, recall, f1, fbeta, tp, fp, fn


def apply_operating_point(probability, valid, base_threshold, max_fraction, min_pixels):
    threshold = float(base_threshold); scores = probability[valid]
    if max_fraction < 1 and scores.size: threshold = max(threshold, float(np.quantile(scores, 1 - max_fraction)))
    return remove_small_components(probability >= threshold, valid, min_pixels), threshold


def evaluate_candidate(items, threshold, max_fraction, min_pixels):
    rows = []
    for city, probability, scene in items:
        prediction, effective = apply_operating_point(probability, scene.valid, threshold, max_fraction, min_pixels)
        precision, recall, f1, fbeta, tp, fp, fn = confusion_metrics(scene.truth, prediction, scene.valid)
        rows.append({"city": city, "precision": precision, "recall": recall, "f1": f1, "fbeta": fbeta,
                     "tp": tp, "fp": fp, "fn": fn, "effective_threshold": effective})
    return {"base_threshold": float(threshold), "max_fraction": float(max_fraction), "min_pixels": int(min_pixels),
            "macro_precision": float(np.mean([r["precision"] for r in rows])), "macro_recall": float(np.mean([r["recall"] for r in rows])),
            "macro_f1": float(np.mean([r["f1"] for r in rows])), "macro_fbeta": float(np.mean([r["fbeta"] for r in rows])),
            "min_precision": float(np.min([r["precision"] for r in rows])), "city_rows": rows}


def select_operating_point(items):
    pooled = np.concatenate([probability[scene.valid] for _, probability, scene in items])
    thresholds = np.unique(np.clip(np.quantile(pooled, np.linspace(.50, .999, cfg.THRESHOLD_GRID_SIZE)), cfg.MIN_THRESHOLD, cfg.MAX_THRESHOLD))
    first = [evaluate_candidate(items, t, fraction, 0) for t in thresholds for fraction in cfg.MAX_POSITIVE_FRACTION_CANDIDATES]
    first.sort(key=lambda r: (r["macro_fbeta"], r["min_precision"]), reverse=True)
    rows = [evaluate_candidate(items, r["base_threshold"], r["max_fraction"], pixels)
            for r in first[:cfg.TOP_OPERATING_CANDIDATES] for pixels in cfg.MIN_COMPONENT_PIXELS_CANDIDATES]
    feasible = [r for r in rows if r["macro_precision"] >= cfg.MIN_VALIDATION_PRECISION and
                r["min_precision"] >= cfg.MIN_PER_CITY_VALIDATION_PRECISION and r["macro_recall"] >= cfg.MIN_VALIDATION_RECALL]
    best = max(feasible or rows, key=lambda r: (r["macro_fbeta"], r["min_precision"], r["macro_precision"]))
    scores = {"Validation_precision": best["macro_precision"], "Validation_min_city_precision": best["min_precision"],
              "Validation_recall": best["macro_recall"], "Validation_F1": best["macro_f1"],
              "Validation_F0.25": best["macro_fbeta"], "Precision_constraint_met": bool(feasible)}
    return best["base_threshold"], best["max_fraction"], best["min_pixels"], scores, best["city_rows"]


def calculate_metrics(city, probability, prediction, scene, training_cities, threshold, effective, max_fraction, min_pixels):
    y, score, pred = scene.truth[scene.valid].astype(np.uint8), probability[scene.valid].astype(np.float32), prediction[scene.valid].astype(np.uint8)
    tp, fp, fn, tn = int(((y == 1) & (pred == 1)).sum()), int(((y == 0) & (pred == 1)).sum()), int(((y == 1) & (pred == 0)).sum()), int(((y == 0) & (pred == 0)).sum())
    total = tp + fp + fn + tn
    return {"Test_city": city, "Training_cities": ",".join(training_cities), "Base_threshold": threshold, "Effective_threshold": effective,
            "Max_positive_fraction": max_fraction, "Min_component_pixels": int(min_pixels),
            "Precision": precision_score(y, pred, zero_division=0), "Recall": recall_score(y, pred, zero_division=0),
            "F1-Score": f1_score(y, pred, zero_division=0), "IoU": tp / (tp + fp + fn) if tp + fp + fn else 0.,
            "Pixel_accuracy": (tp + tn) / total if total else 0., "Specificity": tn / (tn + fp) if tn + fp else 0.,
            "Average_precision": average_precision_score(y, score), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "Valid_pixels": int(scene.valid.sum()), "Reference_positive_pixels": int(y.sum()), "Predicted_positive_pixels": int(pred.sum()),
            "Probability_min": float(score.min()), "Probability_mean": float(score.mean()), "Probability_max": float(score.max())}


def write_raster(path, array, scene, dtype, nodata):
    profile = scene.profile.copy(); profile.update(count=1, dtype=dtype, compress="lzw", nodata=nodata)
    with rasterio.open(path, "w", **profile) as dst: dst.write(array.astype(dtype), 1)
