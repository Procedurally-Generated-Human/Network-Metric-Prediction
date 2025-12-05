# Multi-Horizon Forecasting of Network Metrics Using Machine Learning
This repository is an exploratory final course project for Concordia Univeristy's COMP6321  
Contributors: 
1. Mohammadparsa Toopchinezhad
2. Long Uy Nguyen
3. Daryel Leon Cachott
4. James-Samuel Lemieux-Laing

## Environment
```
python==3.10
pytorch==2.5.1+cu12.4
scikit-learn==1.7.1
matplotlib==3.10.6
pandas==2.3.2
scipy==1.15.2
catboost==1.2.7
numpy==1.26.4
```

## Overview

Multi-horizon forecasting of streaming metrics on Puffer dataset. This project compares four base ML models with basic regression as well as quantile regression
1. Catboost :  `CatBoostRegressor` from `catboost` library 
2. Multi-Layer Perceptron : `MLPRegressor` from `scikit-learn` library, along with custom MLP inherited from pytorch's NN Modules
3. Gated Recurrent Unit : Custom GRU inherited from pytorch's NN module
4. Long-Short Term Memory: Custom LSTM inherited from pytorch's NN module

Algorithms definitions are found in `./src/algorithms.py` file, and all helpers functions are defined within `./src/helpers.py`
All performance comparison plots are found in `./plots/` folder.

## High-level details

The main entry point is within `./src/final.ipynb`, where the Puffer dataset is loaded, along with the necessary parameters. Note that the puffer dataset is a collection of connections, hence consecutive connections may belong to  different groups across different environent. It is therefore necessary to take a only a subset of the dataset such that all samples belong to the same distribution. In our preliminary checks, the first 800000 samples is confirmed to belong to 1 distribution. Base features we are targeting are `['cwnd', 'in_flight', 'min_rtt', 'rtt', 'delivery_rate', 'buffer']`, while the regression target we aim to forecast are: `['rtt' 'delivery_rate']`.

```
    df = helpers.load_puffer_dataset()
    qs = [0.25, 0.75]
    horizons=[1, 2, 4, 8, 16]
    n_lags=10
    sample_size=800000
    base_features = ['cwnd', 'in_flight', 'min_rtt', 'rtt', 'delivery_rate', 'buffer']
```

To run and store our basic regression results + comparison between 4 models:
```
    for metric in ['rtt', 'delivery_rate']:  
        algorithms.run_basic_models_comparison(df, feature_cols=base_features, target_metric=metric, n_lags=5, sample_size=sample_size, horizons=horizons, save_plots=True)
```
To run and store our quantile regression results + comparison between 4 models:
```
for metric in ['rtt', 'delivery_rate']:
    algorithms.run_quantile_models_comparison(df, feature_cols=base_features, target_metric=metric, n_lags=5, sample_size=sample_size, horizons=horizons, qs = qs, save_plots = True)
```
