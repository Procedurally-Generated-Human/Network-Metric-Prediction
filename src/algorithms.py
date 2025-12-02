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
import copy
import torch.nn.functional as F
import helpers
####################### LSTM MODELS ###########################

class LSTM_regressor(nn.Module):
    def __init__(self, in_d, hidden_d, out_d, num_layers, dropout):
        super().__init__()
        self.in_dim = in_d
        self.out_dim = out_d
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=self.in_dim, hidden_size=hidden_d, num_layers=self.num_layers, batch_first=True, dropout=dropout)
        self.dense_out = nn.Linear(hidden_d, self.out_dim)
        
    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        last_output = out[:, -1, :]
        output = self.dense_out(last_output)
        return output
    
class LSTM_quantile(nn.Module):
    def __init__(self, in_d, hidden_d, out_d, num_layers=1, dropout=0.0):
        super().__init__()
        self.in_dim = in_d
        self.hidden_d = hidden_d
        self.num_layers = num_layers
        self.n_q = out_d

        self.lstm = nn.LSTM(
            input_size=self.in_dim,
            hidden_size=hidden_d,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # monotonic head: base (one value) + deltas (n_q-1 positive increments)
        self.head_base = nn.Linear(hidden_d, 1)
        self.head_deltas = nn.Linear(hidden_d, self.n_q - 1)

    def forward(self, x):
        # x: (B, seq_len, in_d)
        out, (h_n, c_n) = self.lstm(x)           # out: (B, seq_len, hidden*dirs)
        last = out[:, -1, :]                     # (B, fc_in)
        base = self.head_base(last)              # (B,1)
        deltas = self.head_deltas(last)          # (B, n_q-1)
        deltas_pos = F.softplus(deltas)          # positive increments (B, n_q-1)
        increments = torch.cumsum(deltas_pos, dim=1)  # cumulative increments (B, n_q-1)
        quantiles_out = torch.cat([base, base + increments], dim=1)  # (B, n_q)
        return quantiles_out

def train_lstm_timeseries(df, target_metric='rtt', n_lags=1, horizon=1, sample_size=800000, model_param={}, visualize=False):
    train_ds, val_ds, test_ds, scaler, target_mean, target_std = helpers.form_puffer_tensor_dataset(
        df, target_metric, n_lags, horizon, sample_size)
    
    model = model_param['model']
    batch_size = model_param['batch_size']
    epochs = model_param['epochs']
    objective = model_param['objective']
    optimizer = model_param['optim']
    DEVICE = model_param['device']
    model.to(DEVICE)
    
    train_losses = []
    val_losses = []
    perf_mae_train = []
    perf_R2_train = []
    perf_mae_val = []
    perf_R2_val = []
    perf_mae_test = []
    perf_R2_test = []
    best_state = None
    patience = 5
    wait = 0
    best_val=float('inf')
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    for epoch in range(epochs):
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)   
        model.train()
        total_loss = 0
        ####### Training #########################
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(x_batch).squeeze(-1)
            loss = objective(y_pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total_loss += loss.item() * x_batch.size(0)
        avg_train = total_loss / len(train_loader.dataset)
        train_losses.append(avg_train)
        
        model.eval()
        #### Checking regression on trainset######
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, drop_last=False)   
        y_pred_train_scaled, y_true_train_scaled = helpers.predict_model(model, train_loader, DEVICE)
        # Manual inverse transform using RTT parameters
        y_pred_lstm = y_pred_train_scaled * target_std + target_mean
        y_true_lstm = y_true_train_scaled * target_std + target_mean
        mae_lstm = mean_absolute_error(y_true_lstm, y_pred_lstm)
        r2_lstm = r2_score(y_true_lstm, y_pred_lstm)
        perf_mae_train.append(mae_lstm)
        perf_R2_train.append(r2_lstm)
        print(f"LSTM MAE (train): {mae_lstm:.6f}, R^2 (train): {r2_lstm:.6f}")
        
        #### On validation set###################
        val_loss = 0
        y_pred = []
        y_true = []
        with torch.no_grad():
            for x_val_batch, y_val_batch in val_loader:
                x_val_batch = x_val_batch.to(DEVICE)
                y_val_batch = y_val_batch.to(DEVICE)
                val_preds = model(x_val_batch).squeeze(-1)
                loss = objective(val_preds, y_val_batch)
                val_loss += loss.item() * x_val_batch.size(0)
                
                # Store predictions and true values (scaled)
                y_pred.append(val_preds.cpu().numpy())
                y_true.append(y_val_batch.cpu().numpy())
        
        # Concatenate predictions and true values
        y_pred_val_scaled = np.concatenate(y_pred, axis=0)
        y_true_val_scaled = np.concatenate(y_true, axis=0)  
        # Manual inverse transform using RTT parameters
        y_pred_lstm = y_pred_val_scaled * target_std + target_mean
        y_true_lstm = y_true_val_scaled * target_std + target_mean
        mae_lstm = mean_absolute_error(y_true_lstm, y_pred_lstm)
        r2_lstm = r2_score(y_true_lstm, y_pred_lstm)
        perf_mae_val.append(mae_lstm)
        perf_R2_val.append(r2_lstm)
        print(f"LSTM MAE (validate): {mae_lstm:.6f}, R^2 (validate): {r2_lstm:.6f}")
        
        #### On test set #############################
        y_pred = []
        y_true = []
        # Concatenate predictions and true values (still scaled)
        y_pred_test_scaled, y_true_test_scaled = helpers.predict_model(model, test_loader, DEVICE)
        # Manual inverse transform using RTT parameters
        y_pred_test = y_pred_test_scaled * target_std + target_mean
        y_true_test = y_true_test_scaled * target_std + target_mean
        mae_lstm = mean_absolute_error(y_true_test, y_pred_test)
        r2_lstm = r2_score(y_true_test, y_pred_test)
        perf_mae_test.append(mae_lstm)
        perf_R2_test.append(r2_lstm)
        print(f"LSTM MAE (test): {mae_lstm:.6f}, R^2 (test): {r2_lstm:.6f}")
        
        # Val Loss ##################
        avg_val = val_loss / len(val_loader.dataset)
        val_losses.append(avg_val)
        print(f" Epoch {epoch:02d} | Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}  \n")
        if avg_val < best_val:
            best_val = avg_val
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait > patience:
                model.load_state_dict(best_state)
                break
    # Return final performance results on last epoch     
    if visualize:
        helpers.visualize_samples(y_true_test, y_pred_test, horizon, target_metric, 200)
        
    data = {'MAE': perf_mae_test[-1], 
            'R2': perf_R2_test[-1], 
            'val_mae': perf_mae_val[-1],
            'val_R2': perf_R2_val[-1],
            'train_mae': perf_mae_train[-1],
            'train_R2': perf_R2_train[-1],
            'train_loss': train_losses[-1],
            'val_loss': val_losses[-1]}
    return model, data

def train_lstm_timeseries_quantile(df, target_metric='rtt', n_lags=1, horizon=1, sample_size=800000, model_param={}, qs=[0.1, 0.25, 0.75, 0.9]):
    train_ds, val_ds, test_ds, scaler, target_mean, target_std = helpers.form_puffer_tensor_dataset(
        df, target_metric, n_lags, horizon, sample_size)
    model = model_param['model']
    batch_size = model_param['batch_size']
    epochs = model_param['epochs']
    objective = model_param['objective']
    optimizer = model_param['optim']
    DEVICE = model_param['device']
    model.to(DEVICE)
    print (f'Training {len(qs)} quantiles, {qs}')
    quantiles = torch.tensor(qs, device=DEVICE, dtype=torch.float32)
    median_idx = helpers.median_index_from_qs(qs)
    pairs = [[qs[i], qs[-(i+1)]] for i in range(len(qs)//2)]
    
    train_losses = []
    val_losses = []
    perf_mae_train = []
    perf_R2_train = []
    perf_mae_val = []
    perf_R2_val = []
    perf_mae_test = []
    perf_R2_test = []
    best_state = None
    patience = 5
    wait = 0
    best_val=float('inf')
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    print("Begin training: ")
    for epoch in range(epochs):
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)   
        model.train()
        total_loss = 0
        ####### Training #########################
        for xb, yb in train_loader:
            xb = xb.to(DEVICE).float()
            yb = yb.to(DEVICE).float()
            optimizer.zero_grad()
            preds = model(xb)                  # (B, n_q)
            loss = objective(preds, yb, quantiles)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_train_loss = total_loss / max(1, len(train_loader.dataset))
        train_losses.append(avg_train_loss)
        
        model.eval()
        #### Checking regression on trainset######
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, drop_last=False)   
        y_pred_train_mat, y_true_train_scaled = helpers.predict_multi_quantiles(model, train_loader, DEVICE)
        # Manual inverse transform using RTT parameters
        y_pred_train_scaled = y_pred_train_mat[:, median_idx]
        y_pred_lstm = y_pred_train_scaled * target_std + target_mean
        y_true_lstm = y_true_train_scaled * target_std + target_mean
        mae_lstm = mean_absolute_error(y_true_lstm, y_pred_lstm)
        r2_lstm = r2_score(y_true_lstm, y_pred_lstm)
        perf_mae_train.append(mae_lstm)
        perf_R2_train.append(r2_lstm)
        print(f"LSTM MAE (train): {mae_lstm:.6f}, R^2 (train): {r2_lstm:.6f}")
        
        #### On validation set###################
        val_loss = 0
        with torch.no_grad():
            for x_val_batch, y_val_batch in val_loader:
                x_val_batch = x_val_batch.to(DEVICE)
                y_val_batch = y_val_batch.to(DEVICE)
                val_preds = model(x_val_batch)
                loss = helpers.quantile_huber_loss(val_preds, y_val_batch, quantiles)
                val_loss += loss.item() * x_val_batch.size(0)

        preds_val_mat, y_true_val_scaled = helpers.predict_multi_quantiles(model, val_loader, DEVICE)
        # Manual inverse transform using RTT parameters
        y_pred_val_scaled = preds_val_mat[:, median_idx]
        y_pred_lstm = y_pred_val_scaled * target_std + target_mean
        y_true_lstm = y_true_val_scaled * target_std + target_mean
        mae_lstm = mean_absolute_error(y_true_lstm, y_pred_lstm)
        r2_lstm = r2_score(y_true_lstm, y_pred_lstm)
        perf_mae_val.append(mae_lstm)
        perf_R2_val.append(r2_lstm)
        print(f"LSTM MAE (validate): {mae_lstm:.6f}, R^2 (validate): {r2_lstm:.6f}")
        
        #### On test set #############################
        # Concatenate predictions and true values (still scaled)
        y_pred_test_mat, y_true_test_scaled = helpers.predict_multi_quantiles(model, test_loader, DEVICE)
        # Manual inverse transform using RTT parameters
        y_pred_test_scaled = y_pred_test_mat[:, median_idx]
        y_pred_test = y_pred_test_scaled * target_std + target_mean
        y_true_test = y_true_test_scaled * target_std + target_mean
        mae_lstm = mean_absolute_error(y_true_test, y_pred_test)
        r2_lstm = r2_score(y_true_test, y_pred_test)
        perf_mae_test.append(mae_lstm)
        perf_R2_test.append(r2_lstm)
        print(f"LSTM MAE (test): {mae_lstm:.6f}, R^2 (test): {r2_lstm:.6f}")
        # Val Loss ##################
        avg_val = val_loss / len(val_loader.dataset)
        val_losses.append(avg_val)
        print(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val:.6f}  \n")
        if avg_val < best_val:
            best_val = avg_val
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait > patience:
                model.load_state_dict(best_state)
                break
            
    preds_test_mat, y_test_scaled = helpers.predict_multi_quantiles(model, test_loader, DEVICE)
    data = {'preds': preds_test_mat,
            'trues': y_test_scaled,
            'target_mean': target_mean,
            'target_std': target_std,
            'quantiles': qs,
            'quantile_pair': pairs}
    return model, data

def run_lstm_comparison_basic(df, model_param, target_metric, n_lags, sample_size, horizons=[1,2,4,8,16]):
    mae_train = []
    mae_val = []
    mae_test = []
    r2_train = []
    r2_val = []
    r2_test = []
    train_loss_final = []
    val_loss_final = []
    for i in horizons:
        model, data = train_lstm_timeseries(df, target_metric=target_metric, n_lags=n_lags, horizon=i, sample_size=sample_size, model_param=model_param, visualize=True)
        mae_test_val = data.get('MAE', None)
        r2_test_val = data.get('R2', None)
        mae_val_val  = data.get('val_mae', None)
        r2_val_val   = data.get('val_R2', None)
        mae_train_val = data.get('train_mae', None)
        r2_train_val  = data.get('train_R2', None)
        train_loss_val = data.get('train_loss', None)
        val_loss_val   = data.get('val_loss', None)
        mae_test.append(float(mae_test_val))
        r2_test.append(float(r2_test_val))
        mae_val.append(float(mae_val_val))
        r2_val.append(float(r2_val_val))
        mae_train.append(float(mae_train_val))
        r2_train.append(float(r2_train_val))
        train_loss_final.append(float(train_loss_val))
        val_loss_final.append(float(val_loss_val))
    
    # Convert to numpy arrays for plotting
    h_arr = np.array(horizons)
    mae_train = np.array(mae_train)
    mae_val = np.array(mae_val)
    mae_test = np.array(mae_test)
    r2_train = np.array(r2_train)
    r2_val = np.array(r2_val)
    r2_test = np.array(r2_test)
    train_loss_final = np.array(train_loss_final)
    val_loss_final = np.array(val_loss_final)
    
    # --- Plot 1: MAE vs Horizon ---
    plt.figure(figsize=(8,5))
    plt.plot(h_arr, mae_train, marker='o', label='MAE (train)')
    plt.plot(h_arr, mae_val, marker='s', label='MAE (val)')
    plt.plot(h_arr, mae_test, marker='^', label='MAE (test)')
    plt.xlabel('Horizon')
    plt.ylabel('MAE')
    plt.title(f'MAE vs Horizon ({target_metric})')
    plt.xticks(h_arr)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # --- Plot 2: R^2 vs Horizon ---
    plt.figure(figsize=(8,5))
    plt.plot(h_arr, r2_train, marker='o', label='R² (train)')
    plt.plot(h_arr, r2_val, marker='s', label='R² (val)')
    plt.plot(h_arr, r2_test, marker='^', label='R² (test)')
    plt.xlabel('Horizon')
    plt.ylabel('R²')
    plt.title(f'R² vs Horizon ({target_metric})')
    plt.xticks(h_arr)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Plot 3: Training Losses vs Horizon (final epoch) ---
    plt.figure(figsize=(8,5))
    plt.plot(h_arr, train_loss_final, marker='o', label='Train loss (final)')
    plt.plot(h_arr, val_loss_final, marker='s', label='Val loss (final)')
    plt.xlabel('Horizon')
    plt.ylabel('Loss (final epoch)')
    plt.title(f'Final Training / Validation Loss vs Horizon ({target_metric})')
    plt.xticks(h_arr)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def run_lstm_comparison_quantile(df, model_param, target_metrics, n_lags, sample_size, horizons=[1,2,4,8,16], qs=[0.1, 0.25, 0.75, 0.9]):
    all_coverages = {}
    all_band_widths = {}
    labels = [ f'{qs[i]*100}%-{qs[-(i+1)]*100}%' for i in range(len(qs)//2) ]
    for i in horizons:
        model, data = train_lstm_timeseries_quantile(df, target_metric=target_metrics, n_lags=n_lags, horizon=i, model_param=model_param, sample_size=sample_size, qs=qs)
        pairs = data['quantile_pair']
        preds = data['preds']
        trues = data['trues']
        target_std = data['target_std']
        target_mean = data['target_mean']
        qs = data['quantiles']
        y_true = trues * target_std + target_mean
        coverages = []
        band_widths = []
        for low, high in pairs:
            low_preds = preds[:, qs.index(low)] * target_std + target_mean
            high_preds = preds[:, qs.index(high)] * target_std + target_mean
            coverage, band_width = helpers.visualize_quantiles(low_preds, high_preds, y_true, low, high, target_metric=target_metrics, horizon=i)
            coverages.append(coverage)
            band_widths.append(band_width)
        all_coverages[f'horizon_{i}'] = coverages
        all_band_widths[f'horizon_{i}'] = band_widths
    helpers.compare_between_horizons(all_coverages, all_band_widths, horizons, labels, target_metrics)
    



##########################################################