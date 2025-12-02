import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import torch
import torch.nn as nn
import random
from torch.utils.data import Dataset, DataLoader, TensorDataset

def load_puffer_dataset():
    """
    Loads the puffer dataset from local CSV file if available, else downloads it
    Returns:
        df: pd.DataFrame 
    """
    try:
        df = pd.read_csv("2025-02-10T11_2025-02-11T11_video_sent_2025-02-10T11_2025-02-11T11.csv")
    except FileNotFoundError:
        response = requests.get("https://storage.googleapis.com/puffer-data-release/2025-02-10T11_2025-02-11T11/video_sent_2025-02-10T11_2025-02-11T11.csv")
        with open("2025-02-10T11_2025-02-11T11_video_sent_2025-02-10T11_2025-02-11T11.csv", 'wb') as f:
            f.write(response.content)
        df = pd.read_csv("2025-02-10T11_2025-02-11T11_video_sent_2025-02-10T11_2025-02-11T11.csv")
    # Step 2: Remove unnecessary columns & rename columns
    #df = df.drop(columns=['session_id', 'channel', 'format'])
    df = df.rename(columns={'time (ns GMT)': 'time'})
    # make time relative to start of session and convert to seconds
    df['time'] = (df['time'] - df['time'].min()) / 1e9
    return df

def form_sliding_windows(df_groups, regression_target, regress_based_on, n_lags, horizon, feature_cols):
    """
    Given grouped views of dataframe, form sliding windows for time series regression tasks based on in regression_based_on column
    Regression_target is the column to be predicted at time t+horizon based on values up to time t

    Args:
        df_groups (pd.DataFrameGroupBy): grouped DataFrame object
        regression_target (string): name of the target regression column (will be added)
        regress_based_on (_type_): regression is based on this column
        n_lags (_type_): number of past timesteps to consider
        horizon (_type_): time horizon to predict ahead
        feature_cols (_type_): set of columns to be used as features

    Returns:
        X_windows: Sliding window feature arrays
        y_windows: Corresponding regression targets
    """
    X_windows = []
    y_windows = []
    for id, group in df_groups:
        g = group.sort_values('time').copy()    # Maintain chronological order
        if regress_based_on not in g.columns: continue
        g = g.reset_index(drop=True)
        g[regression_target] = g[regress_based_on].shift(-horizon)       # Set the target by horizon-lagged regression target --> values up to time t predicts t+horizon value
        g = g.dropna().reset_index(drop=True)
        
        if len(g) < n_lags: continue  # If number of timesteps is less than window length, skip this group
        array = g[feature_cols]             #(B, feature_dim)           # Obtain the values for the focused features
        targets = g[regression_target]      #(B, )                      # Obtain the target regression

        windows = sliding_window_view(array.to_numpy(), window_shape=n_lags, axis=0)
        # Want (B, SEQ_LEN, feature_dim), but sliding_window_view returns (B, feature_dim, SEQ_LEN) for some reason
        if windows.shape[2] != array.shape[1]:
            windows = windows.transpose(0, 2, 1)
        
        X_windows.append(windows)
        y_windows.append(targets.to_numpy()[n_lags-1:])
    X_windows = np.concatenate(X_windows, axis=0)
    y_windows = np.concatenate(y_windows, axis=0)
    assert len(X_windows) == len(y_windows)
    
    return X_windows, y_windows

def quantile_loss(preds, target, qs):
    """
    Compute quantile loss of the current batch
    For y_pred, y_true, quantiles q:
        if y_true >= y_pred: loss = q * (y_true - y_pred)
        else:               loss = (1-q) * (y_pred - y_true)
        
    Args:
        preds:  y_preds (B,k)
        target : y_true (B,)
        qs (_type_): quantiles (k,)

    Returns:
        loss: average quantile loss over the batch
    """
    if not torch.is_tensor(qs):
        qs = torch.tensor(qs, dtype=preds.dtype, device=preds.device)
    else:
        qs = qs.to(preds.dtype).to(preds.device)
    if target.dim() == 1:
        target = target.unsqueeze(1)   # (B,1)
    qs = qs.view(1, -1)               # (1,k) broadcastable
    target_exp = target.expand_as(preds)   # (B,k)
    errors = target_exp - preds            # e = y - y_hat
    loss = torch.max(qs * errors, (qs - 1.0) * errors)   # (B,k)
    return loss.mean()

def quantile_huber_loss(preds, target, qs, delta=0.1):
    """Computes Huber Pinball Quantile Loss
    
    Quantile loss is not smooth at 0, so this adds Huber approximation for stability.
    Huber loss acts on small residuals with a quadratic function, and large residuals with linear function. 
    This allow to be less sensitive to outliers while maintaining differentiability at 0.
    The threshold between small and large residuals is controlled by delta.
    
    Let residual = y_true - y_pred
    if residual >= 0: low predictions 
        if residual < delta: Loss = (q / 2delta) * residual^2   
        else:               Loss = q * residual - 0.5 * q * delta
        
    if residual < 0: high predictions
        if |residual| < delta:  L = ((1-q)/2delta) * residual^2
        else:                   L = (q-1) * residual - 0.5 * (1-q) * delta
        
    Args:
        preds:  y_preds (B,k)
        target : y_true (B,)
        qs: quantiles (k,). 
        delta = Huber threshold, Defaults to 0.1.

    Returns:
        loss: average Huber quantile loss over the batch
    """
    if not torch.is_tensor(qs):
        qs = torch.tensor(qs, dtype=preds.dtype, device=preds.device)
    else:
        qs = qs.to(preds.dtype).to(preds.device)

    if target.dim() == 1:
        target = target.unsqueeze(1)    # (B,1)

    target_exp = target.expand_as(preds)    
    u = target_exp - preds                  
    k = float(delta)
    qs_row = qs.view(1, -1)                 

    # positive branch (u >= 0)
    u_pos = torch.clamp(u, min=0.0)         # (B, n_q)
    pos_small = (u_pos < k)
    loss_pos_small = 0.5 * qs_row * (u_pos ** 2) / k
    loss_pos_large = qs_row * u_pos - 0.5 * qs_row * k
    loss_pos = torch.where(pos_small, loss_pos_small, loss_pos_large)

    # negative branch (u < 0)
    u_neg = torch.clamp(u, max=0.0)         # negative or zero
    neg_small = (u_neg > -k)
    loss_neg_small = 0.5 * (1.0 - qs_row) * (u_neg ** 2) / k
    loss_neg_large = (qs_row - 1.0) * u_neg - 0.5 * (1.0 - qs_row) * k
    loss_neg = torch.where(neg_small, loss_neg_small, loss_neg_large)

    loss = torch.where(u >= 0.0, loss_pos, loss_neg)   # (B, n_q)
    return loss.mean()

def predict_multi_quantiles(model, loader, device):
    """
    Produce multi-quantile predictions using the trained model on the given DataLoader

    Args:
        model: trained Pytorch model
        loader: DataLoader for the dataset to predict on
        device: computation device (CPU/GPU)

    Returns:
        prediction matrix: (N, n_q), rows are samples, columns are predictions for different quantiles
        true values: (N,)
    """
    model.eval()
    preds_list = []
    trues_list = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device).float()
            yp = model(xb)                 # (B, n_q)
            preds_list.append(yp.cpu().numpy())
            trues_list.append(yb.cpu().numpy())
    if len(preds_list) == 0:
        return np.empty((0, model.n_q)), np.empty((0,))
    preds_mat = np.concatenate(preds_list, axis=0)   # (N, n_q)
    y_true = np.concatenate(trues_list, axis=0)      # (N,)
    return preds_mat, y_true

def predict_model(model, data_loader:DataLoader, DEVICE):
    """
    Return y_preds, y_true
    """
    y_preds = []
    y_trues = []
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            y_pred = model(x_batch).squeeze(-1)
            y_preds.append(y_pred.cpu().numpy())
            y_trues.append(y_batch.cpu().numpy())
    y_preds = np.concatenate(y_preds, axis=0)
    y_trues = np.concatenate(y_trues, axis=0)
    return y_preds, y_trues

def median_index_from_qs(qs):
    q_arr = np.array(qs)
    if 0.5 in q_arr:
        return int(np.where(q_arr == 0.5)[0][0])
    return int(np.argmin(np.abs(q_arr - 0.5)))

def form_puffer_tensor_dataset(df:pd.DataFrame, target_metric: str, n_lags: int, horizon: int, sample_size: int=800000):
    print(f"Preparing dataset for target='{target_metric}', lags={n_lags}, Horizon={horizon}")
    df_numeric = df.select_dtypes(include=[np.number]).copy()
    if 'session_id' in df.columns:
        df_numeric['session_id'] = df['session_id']
    if 'time' in df.columns:
        df_numeric['time'] = df['time']
        
    df_numeric = df_numeric.iloc[:sample_size].copy().reset_index(drop=True)
    assert df_numeric['expt_id'].nunique() == 1
    
    # Core features
    base_features = ['cwnd', 'in_flight', 'min_rtt', 'rtt', 'delivery_rate', 'buffer']
    feature_cols = [c for c in base_features if c in df_numeric.columns]
    df_numeric[feature_cols] = df_numeric[feature_cols].astype(np.float32)       
    idx = feature_cols.index(target_metric)                 # Index of the regression value, used to inverse transform the scaled value back to original
    
    df_train, df_test = train_test_split(df_numeric, test_size=0.2, shuffle=False)
    df_train, df_validate = train_test_split(df_train, test_size=0.2, shuffle=False)
    scaler_X = StandardScaler().fit(df_train.loc[:, feature_cols].astype(np.float32))       # Fit scaler on the important columns
    target_mean = scaler_X.mean_[idx]                       # Obtain the scaled parameters for rescaling the regression variable
    target_std = scaler_X.scale_[idx]                       # Each idx holds the parameters of the corresponding idx column
    
    df_train.loc[:, feature_cols] = scaler_X.transform(df_train.loc[:, feature_cols].astype(np.float32))
    df_test.loc[:, feature_cols] = scaler_X.transform(df_test.loc[:,feature_cols].astype(np.float32))
    df_validate.loc[:, feature_cols] = scaler_X.transform(df_validate.loc[:, feature_cols].astype(np.float32))
    regression_target = target_metric+'_next'   # Set the regression target
    
    # Group timeseries by session_id, maintaining the independencies of different connections during window formations
    groups = df_train.groupby('session_id')
    X_train, y_train = form_sliding_windows(groups, regression_target=regression_target, regress_based_on=target_metric, 
                                            n_lags=n_lags, horizon=horizon, feature_cols=feature_cols)
    assert len(X_train) == len(y_train)
    
    groups = df_test.groupby('session_id')
    X_test, y_test = form_sliding_windows(groups, regression_target=regression_target, regress_based_on=target_metric, 
                                          n_lags=n_lags, horizon=horizon, feature_cols=feature_cols)
    assert len(X_test) == len(y_test)
    
    groups = df_validate.groupby('session_id')
    X_val, y_val = form_sliding_windows(groups, regression_target=regression_target, regress_based_on=target_metric, 
                                        n_lags=n_lags, horizon=horizon, feature_cols=feature_cols)
    assert len(X_val) == len(y_val)
    
    print(f"Num training windows : {len(X_train)}, Num validating windows : {len(X_val)}, Num testing windows : {len(X_test)}")
    train_ds = TensorDataset(torch.from_numpy(X_train.astype(np.float32)), torch.from_numpy(y_train.astype(np.float32)))
    val_ds = TensorDataset(torch.from_numpy(X_val.astype(np.float32)), torch.from_numpy(y_val.astype(np.float32)))
    test_ds = TensorDataset(torch.from_numpy(X_test.astype(np.float32)), torch.from_numpy(y_test.astype(np.float32)))
    
    return train_ds, val_ds, test_ds, scaler_X, target_mean, target_std

def visualize_samples(y_true, y_pred, horizon, target_metric, n_samples=200):
    plt.figure(figsize=(10,4))
    plt.plot(y_true[:n_samples], label='Actual', linewidth=2)
    plt.plot(y_pred[:n_samples], label='LSTM prediction', linestyle='--')
    plt.title(f'Next-Step LSTM Prediction (LSTM, {horizon}-step {target_metric}) on test data')
    plt.xlabel('Time Steps')
    plt.ylabel(target_metric)
    plt.legend()
    plt.tight_layout()
    plt.show()

def visualize_quantiles(low_pred, high_pred, y_true, q_low, q_high, horizon, target_metric, n_samples=200):
    plt.figure(figsize=(10,5))
    inside = (y_true >= low_pred) & (y_true <= high_pred)
    coverage = inside.mean() * 100.0  # percentage
    avg_width = np.mean(high_pred - low_pred)
    print(f"Coverage: {coverage:.2f}% | Avg. interval width: {avg_width:.4f}")
    plt.plot(y_true[:n_samples], label='Actual', linewidth=2)
    plt.fill_between(
            np.arange(n_samples),
            low_pred[:n_samples],
            high_pred[:n_samples],
            color='skyblue',
            alpha=0.4,
            label=f'{int(q_low*100)}–{int(q_high*100)}% range'
        )
    plt.plot(low_pred[:200], label='LSTM quantile low', linestyle='--')
    plt.plot(high_pred[:200], label='LSTM quantile high', linestyle='--')
    plt.title(f'Next-Step LSTM Prediction (LSTM, {horizon}-step {target_metric}) on test data')
    plt.xlabel('Time Steps')
    plt.ylabel(f'{target_metric}')
    plt.legend()
    plt.tight_layout()
    plt.show()
    return coverage, avg_width

def compare_between_horizons(all_coverages, all_band_widths, horizons, quantile_ranges, target_metric):
    coverage_array = np.array([all_coverages[f'horizon_{h}'] for h in horizons])
    width_array = np.array([all_band_widths[f'horizon_{h}'] for h in horizons])
    x = np.arange(5)  # horizon positions
    bar_width = 1/len(quantile_ranges)              # 2 bars per horizon
    offsets = [(i - (len(quantile_ranges) - 1) / 2) * bar_width for i in range(len(quantile_ranges))]           
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    for i in range(len(quantile_ranges)):  # 2 quantile ranges
        ax.bar(x + offsets[i], coverage_array[:,i], 
            width=bar_width, label=quantile_ranges[i])
    ax.set_title(f"{target_metric} Coverage Across Horizons")
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Coverage (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([h for h in horizons])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    for i in range(len(quantile_ranges)):  
        ax.bar(x + offsets[i], width_array[:,i], 
            width=bar_width, label=quantile_ranges[i])
    ax.set_title(f"{target_metric} Prediction Interval Width Across Horizons")
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Mean Interval Width")
    ax.set_xticks(x)
    ax.set_xticklabels([h for h in horizons])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()