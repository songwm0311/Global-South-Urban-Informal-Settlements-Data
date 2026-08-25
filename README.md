# File Contents

The repository “Global-South-Urban-Informal-Settlements-Data” provides the datasets, deep-learning model implementation, and analytical codes used to generate global urban informal-settlement maps and assess vulnerability patterns across the Global South.

Repository:
https://github.com/songwm0311/Global-South-Urban-Informal-Settlements-Data

The main contents of the repository are described below.

1. Global South Urban Informal Settlements Data.xlsx

This file contains the primary derived dataset generated in this study. It provides city-level informal-settlement mapping results, uncertainty estimates, socioeconomic characteristics, and vulnerability indicators for 1,914 cities across 98 countries.

The file includes:

(1) Urban and socioeconomic characteristics

UrbanArea: urban extent of each city;
POP: urban population;
Pop.dens: population density;
GDP and per GDP: economic indicators;
Country, CountryName, Continental, and IncomeGroup: geographic and socioeconomic classifications.

These variables support the analysis of the socioeconomic context and regional differences in informal settlements.

(2) Predicted informal-settlement outputs

The dataset provides the derived outputs from the trained deep-learning model, including:

InformalArea: estimated informal-settlement area;
InformalVolume: estimated informal-settlement building volume;
BuildingVolume: total urban building volume;
InformalHeight: estimated informal-settlement building height;
ProInformalArea: proportion of informal settlements within urban areas;
ProInformalVolume: proportion of informal settlements within urban building volume.

These variables are used for the global spatial distribution analysis of informal settlements presented in Fig. 1.

(3) Uncertainty estimates

The dataset includes uncertainty estimates generated through the Monte Carlo Dropout-based uncertainty quantification framework:

Std ProInformalArea: standard deviation of predicted informal-settlement area proportion;
ProInformalArea CI lower: lower bound of the 95% confidence interval;
ProInformalArea CI upper: upper bound of the 95% confidence interval;
Std ProInformalVolume: uncertainty associated with informal-settlement volume proportion.

These variables represent uncertainty propagated from pixel-level settlement prediction to city-level settlement estimates and correspond to the uncertainty assessment described in the manuscript.

(4) Vulnerability assessment variables

The dataset contains all indicators used to construct the Urban Informal Settlements Vulnerability Index (UISVI), including:

Living Conditions Vulnerability (VIS1):
GDP p.c.;
population density;
greening ratio;
water quality;
Vulnerability S1.
Social Infrastructure Vulnerability (VIS2):
electricity availability;
road density;
healthcare accessibility;
school accessibility;
Vulnerability S2.
Environmental Risk Vulnerability (VIS3):
heatwave exposure;
flood risk;
PM2.5 exposure;
NO₂ exposure;
Vulnerability S3.
Integrated vulnerability index:
Vulnerability: composite UISVI value.

These variables correspond to the vulnerability assessment framework and regional vulnerability comparisons presented in Figs. 2–3.

2. Regional Training and Validation Datasets

The repository contains regional sample datasets used for model training, validation, and regional performance evaluation.

The available datasets include:

East Asia.zip
South Asia.zip
Southeast Asia.zip
Middle Asia.zip
West Asia.zip
North Africa.zip
SubSaharan Africa.zip
Middle America.zip
South America.zip

Each regional dataset contains image samples and corresponding labels used for developing and evaluating the urban informal-settlement segmentation model.

These datasets support the construction of training and validation samples described in the machine-learning pipeline.

3. Deep-learning Model and Training Codes

Model architecture

The folder:

models/

contains the deep-learning model architectures used for urban informal-settlement segmentation.

Training workflow

The script:

train.py

implements model training, including model optimization, parameter updating, and training procedures.

Model testing and inference

The scripts:

test_1.py
test_2.py

are used for model evaluation and prediction generation.

The script:

load_model.py

provides functions for loading trained models and performing inference.

Configuration and preprocessing

The following files support model reproduction:

config_all.py: model configuration and hyperparameter settings;
dataloder_Pick.py: image patch loading and preprocessing;
nameList.py: dataset organization and file management.

4. Supporting Computational Functions

The repository also includes supporting codes required for data processing and visualization.

loss/: loss functions used during model optimization;
utils/: auxiliary functions for model training, prediction, and data processing;
PLT_imshow.py: visualization of input images and model outputs;
plt_data.py: data visualization and plotting procedures.

5. Documentation and Execution Records
README.md provides repository descriptions, usage instructions, and workflow information.
nohup.out contains execution logs generated during model runs.
__pycache__/ contains automatically generated Python cache files.
Reproducibility Statement

The repository provides the essential data products and computational workflow required to reproduce the main analyses of this study, including:

generation of urban informal-settlement predictions;
estimation of city-level settlement extent and volume;
Monte Carlo Dropout-based uncertainty quantification;
construction of VIS1, VIS2, VIS3, and UISVI indicators;
regional comparison of informal-settlement vulnerability patterns.

Due to licensing restrictions, original satellite imagery and some third-party auxiliary datasets are not redistributed. However, their data sources, preprocessing procedures, and analytical methods are documented in the manuscript and Supporting Information.
