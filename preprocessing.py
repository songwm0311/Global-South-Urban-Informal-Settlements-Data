"""Raster/vector validation, feature construction, and tiled datasets."""
import random
import warnings
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from scipy import ndimage
import torch
from torch.utils.data import Dataset
import config as cfg


def normalize_reflectance(spectral):
    spectral = np.nan_to_num(spectral.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    finite = spectral[np.isfinite(spectral)]
    if finite.size and float(np.percentile(finite, 99)) > 2.0:
        spectral /= 10000.0
    return np.clip(spectral, 0.0, 1.0).astype(np.float32)


def normalized_difference(a, b, eps=1e-6):
    return np.clip((a - b) / (a + b + eps), -1.0, 1.0).astype(np.float32)


def local_std_map(image, window=cfg.TEXTURE_WINDOW):
    image = image.astype(np.float32)
    mean = ndimage.uniform_filter(image, size=window, mode="reflect")
    mean2 = ndimage.uniform_filter(image * image, size=window, mode="reflect")
    return np.sqrt(np.maximum(mean2 - mean * mean, 0.0)).astype(np.float32)


def uisi_map(swir1, swir2, red_edge4, eps=1e-6):
    """UISI=(SWIR1+SWIR2-2*RE4)/(SWIR1+SWIR2+2*RE4)."""
    numerator = swir1 + swir2 - 2.0 * red_edge4
    denominator = swir1 + swir2 + 2.0 * red_edge4
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float32),
        where=np.abs(denominator) > eps,
    ).astype(np.float32)


def glcm_mean_variance_maps(
    image,
    window=cfg.TEXTURE_WINDOW,
    levels=cfg.GLCM_LEVELS,
    distance=cfg.GLCM_DISTANCE,
    direction=cfg.GLCM_DIRECTION,
):
    """Local GLCM mean and variance from p(m,n|d,s) in the supplied formulas.

    Summing p(m,n|d,s) over n gives the marginal probability of base gray
    level m. Computing that marginal directly is algebraically identical to
    the two double sums and avoids materializing a levels x levels matrix at
    every pixel.
    """
    if levels < 2 or distance < 1:
        raise ValueError("GLCM levels must be >=2 and distance must be >=1")
    dr, dc = int(direction[0]) * distance, int(direction[1]) * distance
    if dr == 0 and dc == 0:
        raise ValueError("GLCM direction cannot be (0, 0)")
    quantized = np.clip(np.floor(image.astype(np.float32) * levels), 0, levels - 1).astype(np.int16)
    pair_base = np.zeros_like(quantized, dtype=bool)
    rows = slice(max(0, -dr), min(quantized.shape[0], quantized.shape[0] - dr))
    cols = slice(max(0, -dc), min(quantized.shape[1], quantized.shape[1] - dc))
    pair_base[rows, cols] = True
    normalizer = ndimage.uniform_filter(pair_base.astype(np.float32), size=window, mode="constant")
    mean = np.zeros_like(image, dtype=np.float32)
    second_moment = np.zeros_like(image, dtype=np.float32)
    for m in range(levels):
        pair_at_m = ((quantized == m) & pair_base).astype(np.float32)
        count = ndimage.uniform_filter(pair_at_m, size=window, mode="constant")
        probability = np.divide(count, normalizer, out=np.zeros_like(count), where=normalizer > 0)
        mean += float(m) * probability
        second_moment += float(m * m) * probability
    variance = np.maximum(second_moment - mean * mean, 0.0)
    scale = float(levels - 1)
    return (mean / scale).astype(np.float32), (variance / (scale * scale)).astype(np.float32)


def window_starts(length, tile_size, stride):
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


class CityScene:
    def __init__(self, city_id, image_path, shp_path):
        self.city_id = str(city_id)
        self.image_path, self.shp_path = image_path, shp_path
        for path in (image_path, shp_path):
            if not path.exists():
                raise FileNotFoundError(f"[{city_id}] Required file not found: {path}")
        with rasterio.open(image_path) as src:
            if src.count != len(cfg.TIFF_BAND_ORDER):
                raise ValueError(f"[{city_id}] TIFF has {src.count} bands, expected {len(cfg.TIFF_BAND_ORDER)}")
            if src.crs is None:
                raise ValueError(f"[{city_id}] TIFF has no CRS")
            descriptions = [d.strip() if isinstance(d, str) else None for d in src.descriptions]
            if all(descriptions) and descriptions != cfg.TIFF_BAND_ORDER:
                raise ValueError(f"[{city_id}] TIFF band descriptions do not match configured order")
            positions = [cfg.TIFF_BAND_ORDER.index(name) + 1 for name in cfg.MODEL_BAND_NAMES]
            self.spectral = normalize_reflectance(src.read(positions).astype(np.float32))
            self.height, self.width = src.height, src.width
            self.transform, self.crs, self.profile = src.transform, src.crs, src.profile.copy()
            self.valid = src.dataset_mask() > 0
            self.valid &= np.all(np.isfinite(self.spectral), axis=0)
            self.valid &= np.any(self.spectral != 0, axis=0)
        labels = gpd.read_file(shp_path)
        labels = labels[labels.geometry.notna() & ~labels.geometry.is_empty].copy()
        if labels.empty or labels.crs is None:
            raise ValueError(f"[{city_id}] Shapefile has no usable geometry or CRS")
        labels = labels.to_crs(self.crs)
        self.truth = rasterize(((g, 1) for g in labels.geometry), out_shape=(self.height, self.width),
                               transform=self.transform, fill=0, dtype="uint8", all_touched=False)
        if int(self.truth[self.valid].sum()) == 0:
            raise ValueError(f"[{city_id}] Labels produce zero positive pixels inside valid raster")
        self.quality_rows, self.band_median, self.band_iqr = [], {}, {}
        for i, name in enumerate(cfg.MODEL_BAND_NAMES):
            values = self.spectral[i][self.valid]
            if not values.size:
                raise ValueError(f"[{city_id}] Band {name} has no valid pixels")
            median, std = float(np.median(values)), float(values.std())
            tolerance = max(1e-7, max(abs(median), 1.0) * 1e-6)
            near_constant = bool(std <= tolerance)
            row = {"City": self.city_id, "Band": name, "Min": float(values.min()),
                   "Median": median, "P99": float(np.percentile(values, 99)),
                   "Max": float(values.max()), "Range": float(values.max() - values.min()),
                   "Std": std, "Constant_tolerance": tolerance, "Near_constant": near_constant}
            self.quality_rows.append(row)
            self.band_median[name] = median
            q25, q75 = np.percentile(values, [25, 75])
            self.band_iqr[name] = max(float(q75 - q25), 1e-4)
            if name in cfg.REQUIRED_NONCONSTANT_BANDS and near_constant:
                message = f"[{city_id}] Required band {name} is nearly constant; regenerate before final reporting"
                if cfg.CONSTANT_BAND_POLICY == "raise":
                    raise ValueError(message)
                if cfg.CONSTANT_BAND_POLICY == "warn":
                    warnings.warn(message, RuntimeWarning)
                else:
                    raise ValueError("CONSTANT_BAND_POLICY must be 'warn' or 'raise'")


def load_scenes(city_files, output_dir):
    scenes = {city: CityScene(city, paths["image"], paths["shp"]) for city, paths in city_files.items()}
    report = pd.DataFrame([row for scene in scenes.values() for row in scene.quality_rows])
    report.to_csv(output_dir / "input_band_quality_report.csv", index=False, encoding="utf-8-sig")
    reliable = report.groupby("Band")["Near_constant"].apply(lambda x: not bool(x.any()))
    active = [name for name in cfg.MODEL_BAND_NAMES if bool(reliable.get(name, False))]
    if "B4" not in active or "B8" not in active:
        raise ValueError("B4 and B8 must vary in every city for NDVI and texture")
    report[report["Near_constant"]].to_csv(output_dir / "near_constant_band_report.csv", index=False, encoding="utf-8-sig")
    return scenes, active


def make_model_input(spectral_tile, scene, active_band_names):
    channel = {name: i for i, name in enumerate(cfg.MODEL_BAND_NAMES)}
    channels = []
    for name in active_band_names:
        robust = np.clip((spectral_tile[channel[name]] - scene.band_median[name]) / scene.band_iqr[name], -5, 5) / 5
        channels.append(robust.astype(np.float32))
    channels.append(normalized_difference(spectral_tile[channel["B8"]], spectral_tile[channel["B4"]]))
    texture = local_std_map(spectral_tile[channel["B8"]])
    channels.append((np.clip(texture / max(scene.band_iqr["B8"], 1e-4), 0, 5) / 5).astype(np.float32))

    # UISI uses Sentinel-2 SWIR1 (B11), SWIR2 (B12), and red-edge 4 (B7).
    channels.append(uisi_map(
        spectral_tile[channel["B11"]],
        spectral_tile[channel["B12"]],
        spectral_tile[channel["B7"]],
    ))

    # The supplied GLCM equations are evaluated on the same B8/NIR source as
    # the pre-existing local texture feature. Both statistics vary per pixel.
    glcm_mean, glcm_variance = glcm_mean_variance_maps(spectral_tile[channel["B8"]])
    channels.extend([glcm_mean, glcm_variance])
    return np.stack(channels).astype(np.float32)


def make_indicator_maps(scene):
    """Return full-scene UISI, GLCM mean, and GLCM variance maps."""
    channel = {name: i for i, name in enumerate(cfg.MODEL_BAND_NAMES)}
    uisi = uisi_map(
        scene.spectral[channel["B11"]],
        scene.spectral[channel["B12"]],
        scene.spectral[channel["B7"]],
    )
    glcm_mean, glcm_variance = glcm_mean_variance_maps(
        scene.spectral[channel["B8"]]
    )
    return uisi, glcm_mean, glcm_variance


def build_city_tile_index(scene, stride, training, seed):
    positive, negative = [], []
    for y in window_starts(scene.height, cfg.TILE_SIZE, stride):
        for x in window_starts(scene.width, cfg.TILE_SIZE, stride):
            y2, x2 = min(y + cfg.TILE_SIZE, scene.height), min(x + cfg.TILE_SIZE, scene.width)
            valid = scene.valid[y:y2, x:x2]
            if float(valid.mean()) < cfg.MIN_VALID_FRACTION:
                continue
            item = (scene.city_id, x, y)
            (positive if int((scene.truth[y:y2, x:x2] * valid).sum()) >= cfg.MIN_POSITIVE_PIXELS else negative).append(item)
    if not positive:
        raise ValueError(f"[{scene.city_id}] No positive training tiles")
    if training:
        maximum = int(np.ceil(len(positive) * cfg.NEGATIVE_TO_POSITIVE_RATIO))
        if len(negative) > maximum:
            selected = np.random.default_rng(seed).choice(len(negative), size=maximum, replace=False)
            negative = [negative[i] for i in selected]
    return positive + negative


class MultiCityTileDataset(Dataset):
    def __init__(self, scenes, city_ids, active_band_names, training=True, seed=42):
        self.scenes, self.active, self.training, self.items = scenes, active_band_names, training, []
        for offset, city in enumerate(city_ids):
            self.items.extend(build_city_tile_index(scenes[city], cfg.TRAIN_STRIDE, training, seed + offset))
    def __len__(self): return len(self.items)
    def __getitem__(self, index):
        city, x, y = self.items[index]; scene = self.scenes[city]
        y2, x2 = min(y + cfg.TILE_SIZE, scene.height), min(x + cfg.TILE_SIZE, scene.width)
        spectral, truth, valid = scene.spectral[:, y:y2, x:x2], scene.truth[y:y2, x:x2], scene.valid[y:y2, x:x2]
        h, w = truth.shape
        if h < cfg.TILE_SIZE or w < cfg.TILE_SIZE:
            ph, pw = cfg.TILE_SIZE - h, cfg.TILE_SIZE - w
            spectral = np.pad(spectral, ((0, 0), (0, ph), (0, pw)), mode="edge")
            truth = np.pad(truth, ((0, ph), (0, pw)), mode="constant")
            valid = np.pad(valid, ((0, ph), (0, pw)), mode="constant")
        features, truth, valid = make_model_input(spectral, scene, self.active), truth.astype(np.float32), valid.astype(np.float32)
        if self.training:
            if random.random() < .5: features, truth, valid = np.flip(features, 2).copy(), np.flip(truth, 1).copy(), np.flip(valid, 1).copy()
            if random.random() < .5: features, truth, valid = np.flip(features, 1).copy(), np.flip(truth, 0).copy(), np.flip(valid, 0).copy()
            k = random.randint(0, 3)
            if k: features, truth, valid = np.rot90(features, k, (1, 2)).copy(), np.rot90(truth, k).copy(), np.rot90(valid, k).copy()
        return torch.from_numpy(features), torch.from_numpy(truth[None]), torch.from_numpy(valid[None])
