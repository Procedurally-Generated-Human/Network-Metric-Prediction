# Multi-Horizon Forecasting of Network Metrics Using Machine Learning
This repository is an exploratory final course project for Concordia Univeristy's COMP6321: Machine Learning. The full report can be read [here](https://github.com/Procedurally-Generated-Human/Network-Metric-Prediction/blob/main/Final-Report.pdf).

Contributors: 
1. Mohammadparsa Toopchinezhad
2. Long Uy Nguyen
3. Daryel Leon Cachott
4. James-Samuel Lemieux-Laing

## Abstract

Accurate forecasting of network metrics such as throughput and delay is essential
for adaptive bitrate (ABR) algorithms, which rely on these predictions to make
stable rate-selection decisions and maintain user Quality of Experience (QoE)
during video streaming. While most prior work focuses on single-step forecasting,
multi-horizon predictions can help ABRs anticipate future network changes rather
than react to them. We study multi-horizon forecasting of key network metrics using
real-world data, evaluating CatBoost, MLP, GRU, and LSTM models across several
horizons. We also incorporate quantile regression via the pinball loss to capture
uncertainty in highly variable network conditions. Our results show that treebased models maintain the strongest point-forecast accuracy across horizons, while
recurrent models offer competitive short-term performance. Quantile regression
produces well-calibrated uncertainty intervals whose width grows appropriately
with horizon, providing more informative predictions for downstream control.

![Visual Abstract](https://github.com/Procedurally-Generated-Human/Network-Metric-Prediction/blob/main/Visual-Abstract.png)

## Sample Plots
![Quantile](https://github.com/Procedurally-Generated-Human/Network-Metric-Prediction/blob/main/plots/rtt_horizon4_lstm_25.0_75.0_quantreg.png)
![Act-vs-Pred](https://github.com/Procedurally-Generated-Human/Network-Metric-Prediction/blob/main/plots/delivery_rate_horizon_2_basic_regression.png)
![Comparison](https://github.com/Procedurally-Generated-Human/Network-Metric-Prediction/blob/main/plots/rtt_test_maes.png)

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
2. Multi-Layer Perceptron : `MLPRegressor` from `scikit-learn` library, along with custom MLP definition, based inherited from pytorch's NN Modules
3. Gated Recurrent Unit : Custom GRU definition, base inherited from pytorch's NN module
4. Long-Short Term Memory: Custom LSTM definition, base inherited from pytorch's NN module

Algorithms definitions are found in `./src/algorithms.py` file, and all helpers functions are defined within `./src/helpers.py`
All performance comparison plots are found in `./plots/` folder.

## High-level details

The main entry point is within `./src/final.ipynb`, where the Puffer dataset is loaded, along with the necessary parameters. Note that the Puffer dataset is a collection of real connections, hence consecutive rows may belong to different groups across different environent. It is therefore necessary to take a only a subset of the dataset such that all samples belong to the same distribution. In our preliminary checks, the first 800000 samples is confirmed to belong to 1 distribution. Base features we are targeting are `['cwnd', 'rtt', 'delivery_rate', 'buffer']`, while the regression target we aim to forecast are: `['rtt' 'delivery_rate']`. Note that we're trying to forecast the next timestep's value. Internally, lagged features + windows are formed with respect to `session_id`.

```
    df = helpers.load_puffer_dataset()
    qs = [0.25, 0.75]
    horizons=[1, 2, 4, 8, 16]
    n_lags=10
    sample_size=800000
    base_features = ['cwnd','rtt', 'delivery_rate', 'buffer']
    df = helpers.smooth_dataset(df, smoothing_window_size=10, keep_orig=False, groupby='session_id', feature_cols=base_features)
```

To run and store our basic regression results + comparison between 4 models:
```
for metric in ['rtt', 'delivery_rate']:  
    algorithms.run_basic_models_comparison(df, feature_cols=base_features, target_metric=metric, n_lags=n_lags, sample_size=sample_size, horizons=horizons, save_plots=True)
```
To run and store our quantile regression results + comparison between 4 models:
```
for metric in ['rtt', 'delivery_rate']:
    algorithms.run_quantile_models_comparison(df, feature_cols=base_features, target_metric=metric, n_lags=n_lags, sample_size=sample_size, horizons=horizons, qs=qs, save_plots=True)
```


## Citation

If you use this work in your research or projects, please cite:

### 📎 BibTeX

```bibtex
@misc{toopchinezhad2025multihorizon,
  title        = {Multi-Horizon Forecasting of Network Metrics Using Machine Learning},
  author       = {Toopchinezhad, Mohammadparsa and Nguyen, Long Uy and Cachott, Daryel Leon and Lemieux-Laing, James-Samuel},
  year         = {2025},
  howpublished = {Final Course Project, COMP 6321: Machine Learning, Concordia University},
  url          = {https://github.com/Procedurally-Generated-Human/Network-Metric-Prediction}
}
