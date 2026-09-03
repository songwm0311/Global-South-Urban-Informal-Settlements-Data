# Global South Urban Informal Settlements — processing code

This code-only deposit contains the U-Net preprocessing, training, inference,
nested out-of-fold calibration, independent evaluation, and visualization workflow supplied for the study. 


## Input manifest

Copy `data_manifest.csv` and add one row per available city:

```text
city_id,image_path,label_path
<city identifier>,<10-band Sentinel-2 GeoTIFF>,<reference polygon Shapefile>
```

Paths may be absolute or relative to the manifest. The code does not prescribe
or include any regional sample names. A Shapefile must retain its normal
companion files (`.dbf`, `.shx`, `.prj`, and, where applicable, `.cpg`) in the
same directory.

## Code organization

- `models/`: U-Net architecture.
- `train.py`: train and independently evaluate one manifest-selected outer fold.
- `test_1.py`: load one checkpoint and generate a probability GeoTIFF.
- `test_2.py`: run every manifest city as an independent outer test fold.
- `load_model.py`: checkpoint loading.
- `config_all.py`: preprocessing and model hyperparameters.
- `dataloder_Pick.py`: patch loader and preprocessing interface (filename retained from the supplied description).
- `nameList.py`: manifest validation and path management.
- `loss/`: false-positive-aware BCE plus Tversky loss.
- `utils/`: evaluation, raster output, UISI, and GLCM helpers.
- `PLT_imshow.py`: raster visualization.
- `plt_data.py`: city-metric visualization.
- `workflow.py`, `preprocessing.py`, `training.py`, and `evaluation.py`: separated workflow implementation.

## UISI and GLCM formulas

UISI is calculated per pixel as:

`UISI = (B11 + B12 - 2*B7) / (B11 + B12 + 2*B7)`

where B11, B12, and B7 represent SWIR1, SWIR2, and red-edge band 4. GLCM mean
and variance follow the supplied probability-moment formulas and are calculated
as local B8 texture maps. Because the supplied formula did not specify gray
levels `k`, direction `d`, or distance `s`, these are explicit settings in
`config_all.py` (defaults: 32 levels, one-pixel distance, 0-degree direction).

## Installation and execution

The supplied notebook reports Python 3.10.19. Install the recorded direct
dependency versions in a clean Python 3.10 environment:

```text
python -m venv .venv
.venv/Scripts/activate
python -m pip install -r requirements.txt
```

Train and evaluate one outer fold:

```text
python train.py --manifest /path/to/manifest.csv --test-city CITY_ID --output-dir results
```

Run all manifest cities as outer test folds:

```text
python test_2.py --manifest /path/to/manifest.csv --output-dir results
```

Run inference with one trained checkpoint:

```text
python test_1.py --manifest /path/to/manifest.csv --city CITY_ID --checkpoint MODEL.pth --output probability.tif
```

## Outputs

For each independent test city the code writes model checkpoints, training
history, OOF validation details, final metrics, run metadata, ensemble
probability and standard deviation, raw and filtered predictions, ground truth,
UISI, GLCM mean, and GLCM variance GeoTIFFs.

## Scope of reproducibility

This archive implements the processing present in the supplied U-Net notebook
plus the subsequently supplied UISI and GLCM formulas. The broader repository
description additionally names city-level area/volume aggregation, Monte Carlo
Dropout uncertainty, VIS1–VIS3, and UISVI. Their complete equations and source
code were not present in the supplied notebook or formula image, so they are
not invented here. The ensemble-member standard deviation produced by this
workflow is not labeled as Monte Carlo Dropout uncertainty.



