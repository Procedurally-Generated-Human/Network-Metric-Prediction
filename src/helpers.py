import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from numpy.lib.stride_tricks import sliding_window_view
import torch
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
    Given grouped views of dataframe, form sliding windows for time series regression tasks based on `regression_based_on` column  
    `Regression_target` is the column to be predicted at time `t+horizon` based on values up to time t
    
    This assumes df_groups are selected from the same distribution

    Args:
        df_groups (pd.DataFrameGroupBy): grouped DataFrame object
        regression_target (string): name of the target regression column (will be added)
        regress_based_on (_type_): regression is based on this column
        n_lags (_type_): number of past timesteps to consider
        horizon (_type_): timestep to predict ahead
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
        array = g.loc[:,feature_cols].to_numpy()             #(B, feature_dim)           # Obtain the values for the focused features
        targets = g[regression_target].to_numpy()      #(B, )                      # Obtain the target regression

        windows = sliding_window_view(array, window_shape=n_lags, axis=0)
        # Want (B, SEQ_LEN, feature_dim), but sliding_window_view returns (B, feature_dim, SEQ_LEN) for some reason
        if windows.shape[2] != array.shape[1]:
            windows = windows.transpose(0, 2, 1)
        window_count = windows.shape[0]
        indices = np.arange(n_lags-1, n_lags-1 + window_count)
        targets = targets[indices]
        
        X_windows.append(windows)
        y_windows.append(targets)
    X_windows = np.concatenate(X_windows, axis=0)
    y_windows = np.concatenate(y_windows, axis=0)
    assert len(X_windows) == len(y_windows)
    return X_windows, y_windows

def make_lagged_dataset(df_groups, target, n_lags, horizon, feature_cols):
    """
    Given grouped views of dataframe, form lagged features for time series regression task
    `Regression_target` is the column to be predicted at time t+horizon based on values up to time t
    
    This assumes df_groups are selected from the same distribution
    
    Args:
        df_groups (pd.DataFrameGroupBy): grouped DataFrame object
        target (string): name of the target regression column
        n_lags (_type_): number of past timesteps to consider
        horizon (_type_): timestep to predict ahead
        feature_cols (_type_): set of columns to be used as base features

    Returns:
        X_lagged: Rows with lagged features
        y_lagged: Corresponding regression targets
    """
    X_lagged = []
    y_lagged = []
    for sid, g in df_groups:
        g = g.sort_values("time").copy()
        g = g[feature_cols]
        # Add lag features
        if len(g) < n_lags+1: continue  # If number of timesteps is less than window length, skip this group
        
        for lag in range(1, n_lags):
            for c in feature_cols:
                g[f"{c}_lag{lag}"] = g[c].shift(lag)

        g[f"{target}_next{horizon}"] = g[target].shift(-horizon)
        g = g.dropna().reset_index(drop=True)
        # Select feature columns (lags + raw)
        X_lagged.append(g.iloc[:, :-1])
        y_lagged.append(g.iloc[:, -1])
    
    X_lagged = np.concatenate(X_lagged, axis=0)
    y_lagged = np.concatenate(y_lagged, axis=0)
    return X_lagged, y_lagged

def quantile_loss(pred, target, q):
    e = target - pred
    return torch.mean(torch.max(q*e, (q-1)*e))

def predict_multi_quantiles(model, loader, device):
    """
    Produce multi-quantile predictions using the trained LSTM model on the given DataLoader

    Args:
        model: trained Pytorch model
        loader: DataLoader for the dataset to predict on
        device: computation device (CPU/GPU)

    Returns:
        prediction matrix: (N, n_q), rows are samples, columns are predictions for different quantiles
        true values: (N,)
    """
    model.to(device)
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
    Return y_preds, y_true of the current model, predicting on the data_loader
    """
    model.eval()   
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

def form_puffer_tensor_dataset(df_train:pd.DataFrame, df_test:pd.DataFrame, feature_cols: list[str], target_metric: str, n_lags: int, horizon: int):
    """
    Form TensorDataset for the raw MTS data, after forming sliding windows

    Args:
        df_train (pd.DataFrame): train MTS
        df_test ( pd.Dataframe): test MTS
        feature_cols (list[str]): column names of important features
        target_metric (str): column name of regression target
        n_lags (int): number of timesteps lagged
        horizon (int): number of timesteps ahead to predict
    """
    regression_target = target_metric+'_next'   # Set the regression target
    
    # Group timeseries by session_id, maintaining the independencies of different connections during window formations
    groups = df_train.groupby('session_id', sort=False)
    X_train, y_train = form_sliding_windows(groups, regression_target=regression_target, regress_based_on=target_metric, 
                                            n_lags=n_lags, horizon=horizon, feature_cols=feature_cols)
    assert len(X_train) == len(y_train)
    
    groups = df_test.groupby('session_id', sort=False)
    X_test, y_test = form_sliding_windows(groups, regression_target=regression_target, regress_based_on=target_metric, 
                                          n_lags=n_lags, horizon=horizon, feature_cols=feature_cols)
    assert len(X_test) == len(y_test)
    
    print(f"Num training windows : {len(X_train)}, Num testing windows : {len(X_test)}")
    train_ds = TensorDataset(torch.from_numpy(X_train.astype(np.float32)), torch.from_numpy(y_train.astype(np.float32)))
    test_ds = TensorDataset(torch.from_numpy(X_test.astype(np.float32)), torch.from_numpy(y_test.astype(np.float32)))
    
    return train_ds, test_ds

def visualize_samples(y_true, y_pred, horizon, target_metric, n_samples=200):
    """
    Display target_metric's y_true vs y_pred for a horizon, for n_samples
    """
    plt.figure(figsize=(10,4))
    plt.plot(y_true[:n_samples], label='Actual', linewidth=2)
    plt.plot(y_pred[:n_samples], label='LSTM prediction', linestyle='--')
    plt.title(f'Next-Step LSTM Prediction (LSTM, {horizon}-step {target_metric}) on test data')
    plt.xlabel('Time Steps')
    plt.ylabel(target_metric)
    plt.legend()
    plt.tight_layout()
    plt.show()

def visualize_quantiles(low_pred, high_pred, y_true, q_low, q_high, horizon, target_metric, n_samples=200, model_name="",save_plot=False, path=""):
    """
    Display target_metric's quantile-pair prediction, for n_samples.  
    Returns coverage, avg_width of the quantile predictions
    
    Args:
        low_pred (np.ndarray) : low quantile prediction
        high_pred (np.ndarray) : high quantile prediction
        y_true (np.ndarray) : true values
        q_low (float) : Low quantile percentage value
        q_high (float) : High quantile percentage value
        horizon (int) : Prediction timestep, for labelling purposes
        target_metric (str) : Regression name, for labelling purposes
        n_samples (int) : number of predictions to display
        model_name (str) : Model name, for labelling purposes
        save_plot (bool) : Toggle saving the plots
        path (str) : Path to save plots
        
    Return:
        coverage (float): average percentage of y_true between [low, high] quantile predictions  
        avg_width (float): average width between [low, high] prediction
    """
    plt.figure(figsize=(12,4))
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
    plt.plot(low_pred[:200], label=f'{q_low} quantile', linestyle='--')
    plt.plot(high_pred[:200], label=f'{q_high} quantile high', linestyle='--')
    plt.title(f'Next-Step Prediction ({model_name}, {horizon}-step {target_metric}) on test data')
    plt.xlabel('Time Steps')
    plt.ylabel(f'{target_metric}')
    plt.legend()
    if save_plot: 
        plt.savefig(path)
    plt.tight_layout()
    plt.show()
    return coverage, avg_width
    
def smooth_dataset(df:pd.DataFrame, smoothing_window_size, keep_orig=False, groupby = None, feature_cols=[]):
    """
    Smooth `feature_cols` the MTS dataset, based on `smoothing_window_size`.  Optionally discard original raw columns

    Args:
        df (pd.DataFrame): MTS dataset
        smoothing_window_size (_type_): length of window to smooth based on
        keep_orig (bool, optional): Toggle whether to keep the default raw values
        groupby (_type_, optional): column name to perform smoothing based on groupby groups.
        feature_cols (list, optional): Columns requiring smoothing.
    """
    def _smooth_series(s:pd.Series):
        return s.rolling(window=smoothing_window_size, min_periods=1, center=False).mean()
    
    df = df.copy()
    if isinstance(groupby, str):
        groups = df.groupby(groupby, sort=False)
        for col in feature_cols:
            smoothed = groups[col].transform(_smooth_series)
            if keep_orig:
                df[f'{col}_smoothed'] = smoothed
            else:
                df[col] = smoothed
    else:
        for col in feature_cols:
            smoothed = _smooth_series(df[col])
            if keep_orig:
                df[f'{col}_smoothed'] = smoothed
            else:
                df[col] = smoothed      
    return df