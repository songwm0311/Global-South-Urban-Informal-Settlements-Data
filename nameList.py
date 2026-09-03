"""Read an external dataset manifest without embedding regional sample names."""
from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = ("city_id", "image_path", "label_path")


def load_city_files(manifest_path, data_root=None):
    manifest_path = Path(manifest_path).resolve()
    frame = pd.read_csv(manifest_path, dtype={"city_id": str})
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing: raise ValueError(f"Manifest is missing columns: {missing}")
    if frame.empty: raise ValueError("Manifest contains no data rows")
    if frame["city_id"].duplicated().any(): raise ValueError("city_id values must be unique")
    root = Path(data_root).resolve() if data_root else manifest_path.parent
    def resolve(value):
        path = Path(value); return path if path.is_absolute() else root / path
    return {str(row.city_id): {"image": resolve(row.image_path), "shp": resolve(row.label_path)}
            for row in frame.itertuples(index=False)}

