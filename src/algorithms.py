import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import torch
import torch.nn as nn
import random
from torch.utils.data import Dataset, DataLoader, TensorDataset
import copy
import torch.nn.functional as F
import helpers
from catboost import CatBoostRegressor, Pool
from sklearn.neural_network import MLPRegressor


####################### LSTM MODELS ###########################

class LSTM_regressor(nn.Module):
    """
    Basic LSTM regressor module

    Args:
        in_d (int): input dimension
        hidden_d (int): hidden layer dimension
        out_d (int): Output dimension
        num_layers (int): Number of hidden layers
        dropout (float): Droput ratio between layers
    """
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
    """
    (THIS IS A PROTOTYPE)
    A LSTM regression model that predicts quantile values. It includes a monotonic head to generate quantile 
    predictions with positive increments. 

    Attributes:
        in_dim (int): input_dim.
        hidden_d (int): hidden units in the LSTM layers.
        num_layers (int): The number of LSTM layers.
        n_q (int): The number of quantiles to predict.
        lstm (nn.LSTM): The LSTM layer used for temporal feature extraction.
        head_base (nn.Linear): A linear layer to predict the base quantile value.
        head_deltas (nn.Linear): A linear layer to predict the positive deltas for the quantiles.
        
    Args:
        in_d (int): input dimension
        hidden_d (int): hidden layer dimension
        out_d (int): Output dimension
        num_layers (int): Number of hidden layers
        dropout (float): Droput ratio between layers
    """
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

class QuantileHuberLoss(nn.Module):
    """
    Custom Quantile Loss integrated with Huber Loss

    Args:
        _qs (list[float]): list of quantiles in monotonic ascending order
        delta (float): Controls the smoothness of the loss function around zero residual. Larger delta = more smoothing
    """
    def __init__(self, qs, delta: float = 0.1):
        super().__init__()
        if not torch.is_tensor(qs):
            qs = torch.tensor(qs, dtype=torch.float32)
        self.register_buffer("_qs", qs.view(-1))  # stored as buffer so moves to device with model
        self.delta = float(delta)

    def forward(self, preds: torch.Tensor, target: torch.Tensor):
        qs = self._qs.to(preds.dtype)

        # normalize shapes
        if target.dim() == 1:
            target = target.unsqueeze(1)        # (B,1)
        # allow preds shaped (B,) -> (B,1)
        if preds.dim() == 1:
            preds = preds.unsqueeze(1)

        # preds: (B, k), target: (B,1)
        # ensure qs shape (1, k) for broadcasting
        qs_row = qs.view(1, -1).to(preds.device, preds.dtype)  # (1, k)

        target_exp = target.expand_as(preds)   # (B, k)
        u = target_exp - preds                 # signed residuals (y - y_hat), shape (B,k)
        k = float(self.delta)

        # positive branch (u >= 0)
        u_pos = torch.clamp(u, min=0.0)
        pos_small = (u_pos < k)
        loss_pos_small = 0.5 * qs_row * (u_pos ** 2) / k
        loss_pos_large = qs_row * u_pos - 0.5 * qs_row * k
        loss_pos = torch.where(pos_small, loss_pos_small, loss_pos_large)

        # negative branch (u < 0)
        u_neg = torch.clamp(u, max=0.0)
        neg_small = (u_neg > -k)
        loss_neg_small = 0.5 * (1.0 - qs_row) * (u_neg ** 2) / k
        loss_neg_large = (qs_row - 1.0) * u_neg - 0.5 * (1.0 - qs_row) * k
        loss_neg = torch.where(neg_small, loss_neg_small, loss_neg_large)

        loss = torch.where(u >= 0.0, loss_pos, loss_neg)  # (B, k)
        return loss

class QuantileLoss(nn.Module):
    """
    Basic Pinball Loss for quantile regression

    Args:
        nn (_type_): _description_
    """
    def __init__(self, qs):
        """
        Quantile loss function for regression tasks.
        
        Args:
            qs (list or tensor): List or tensor of quantile values (e.g., [0.1, 0.5, 0.9])
        """
        super().__init__()
        if not torch.is_tensor(qs):
            qs = torch.tensor(qs, dtype=torch.float32)
        self.register_buffer("_qs", qs.view(-1))  # Store as buffer so it moves with the model to the device

    def forward(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Compute the quantile loss for a given batch of predictions and targets.
        
        Args:
            preds (Tensor): Predicted values from the model (B, k) where k is the number of quantiles.
            target (Tensor): True values (B,) or (B, 1)
            
        Returns:
            Tensor: Quantile loss (B, k)
        """
        qs = self._qs.to(preds.dtype)
        # Normalize shapes: make sure target is (B, 1) and preds are (B, k)
        if target.dim() == 1:
            target = target.unsqueeze(1)  # (B, 1)
        if preds.dim() == 1:
            preds = preds.unsqueeze(1)   # (B, 1)
            
        qs_row = qs.view(1, -1).to(preds.device, preds.dtype)  # (1, k)
        # Expand target to match preds shape: (B, k)
        target_exp = target.expand_as(preds)   # (B, k)
        u = target_exp - preds                 # Residuals (y - y_hat), shape (B, k)

        # Compute quantile loss
        loss = torch.where(u >= 0.0, qs_row * u, (qs_row - 1.0) * u)  # (B, k)
        return loss.mean()

def train_lstm_timeseries(train_ds, test_ds, model_param={}):   
    """
    Trains the LSTM regression model, using default Huberloss

    Args:
        train_ds (TensorDataset): Train Tensor
        test_ds(TensorDataset): Test tensor
        model_param (dict, optional): Model parameters. Defaults to {}.
            Example Parms :
                model_param = {
                    'model': model_lstm,
                    'batch_size': 256,
                    'epochs': 100,
                    'objective': objective,
                    'optim': torch.optim.AdamW(model_lstm.parameters(), lr=1e-1, weight_decay=1e-4),
                    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
                }
    Returns:
        model: trained model  
        metrics (dict): performances  s
            {"train_mae": mean_absolute_error(y_train, pred_train),  
            "train_R2":  r2_score(y_train, pred_train),  
            "MAE":  mean_absolute_error(y_test, pred_test),  
            "R2":   r2_score(y_test, pred_test),  
            "y_pred_test": pred_test,}            
        
    """
    model = model_param['model']
    batch_size = model_param['batch_size']
    epochs = model_param['epochs']
    objective = model_param['objective']
    optimizer = model_param['optim']
    DEVICE = model_param['device']
    model.to(DEVICE)
    
    train_losses = []
    best_state = None
    patience = 5
    wait = 0
    best_val=float('inf')
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)   
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
        print(f" Epoch {epoch+1}/{epochs} | Train Loss: {avg_train:.6f}")
        if avg_train < best_val - 5e-4:
            best_val = avg_train
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait > patience:
                model.load_state_dict(best_state)
                break
    pred_train, y_train = helpers.predict_model(model, train_loader, DEVICE)
    pred_test, y_test = helpers.predict_model(model, test_loader, DEVICE)
    metrics = {
        "train_mae": mean_absolute_error(y_train, pred_train),
        "train_R2":  r2_score(y_train, pred_train),
        "MAE":  mean_absolute_error(y_test, pred_test),
        "R2":   r2_score(y_test, pred_test),
        "y_pred_test": pred_test,
    }
    return model, metrics

def train_lstm_quantiles(train_ds, test_ds, model_param={}):
    """
    Trains the LSTM quantile regression model, using the specified objective within model_param

    Args:
        train_ds (TensorDataset): Train Tensor
        test_ds(TensorDataset): Test tensor
        model_param (dict, optional): Model parameters. Defaults to {}.
            Sample Params:
                    model_param = {
                        'model': model_lstm,
                        'batch_size': 256,
                        'epochs': 100,
                        'objective': QuantileLoss(quantiles),
                        'optim': torch.optim.AdamW(model_lstm.parameters(), lr=1e-1, weight_decay=1e-4),
                        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    }
    Returns:
        model: trained model  
        data (dict): preds are matrices of prediction, each column is a prediction for a quantile: 
            data = {'preds': preds_test_mat,
                    'trues': y_test_scaled,
                    'train_loss': train_losses,}
    """
    model = model_param['model']
    batch_size = model_param['batch_size']
    epochs = model_param['epochs']
    objective = model_param['objective']
    optimizer = model_param['optim']
    DEVICE = model_param['device']
    model.to(DEVICE)
    objective.to(DEVICE)
    
    train_losses = []
    val_losses = []
    best_state = None
    patience = 5
    wait = 0
    best_val=float('inf')
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    print("Begin training: ")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)   
        
        ####### Training #########################
        for xb, yb in train_loader:
            xb = xb.to(DEVICE).float()
            yb = yb.to(DEVICE).float()
            optimizer.zero_grad()
            preds = model(xb)                  # (B, n_q)
            loss = objective(preds, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_train_loss = total_loss / max(1, len(train_loader.dataset))
        train_losses.append(avg_train_loss)
        
        model.eval()        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f}")
        if avg_train_loss < best_val - 5e-4:
            best_val = avg_train_loss
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
            'train_loss': train_losses,
            'val_loss': val_losses,}
    return model, data


################## CatBoost ###################################
def train_catboost_time_series(X_train, y_train, X_test, y_test, model:CatBoostRegressor):
    """
    Trains the catboost model

    Args:
        X_train (NDarray):
        y_train (NDarray): 
        X_test (NDarray):
        y_test (NDarray):
        model (CatBoostRegressor): 

    Returns:
        model (CatBoostRegressor): trained model  
        metrics (dict): performances
            {"train_mae": mean_absolute_error(y_train, pred_train),  
            "train_R2":  r2_score(y_train, pred_train),  
            "MAE":  mean_absolute_error(y_test, pred_test),  
            "R2":   r2_score(y_test, pred_test),  
            "y_pred_test": pred_test,}      
    """
    model = model.fit(Pool(X_train, y_train), eval_set=Pool(X_test, y_test), use_best_model=True)
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    metrics = {
        "train_mae": mean_absolute_error(y_train, pred_train),
        "train_R2":  r2_score(y_train, pred_train),
        "MAE":  mean_absolute_error(y_test, pred_test),
        "R2":   r2_score(y_test, pred_test),
        "y_pred_test": pred_test,
    }
    return model, metrics

def train_catboost_quantile(X_train, y_train, X_test, y_test, quantiles=[0.1, 0.25, 0.75, 0.9]):
    """
    Trains the catboost model through quantile regression. Also compute coverage percentage through the quantile pairs and average width

    Args:
        X_train (NDarray):
        y_train (NDarray): 
        X_test (NDarray):
        y_test (NDarray):
        quantiles (list[float]): list of quantiles to train. Must be monotonic increasing

    Returns:
        model (CatBoostRegressor): trained model  
        data (dict): performances : 
            {"preds": mean_absolute_error(y_train, pred_train),  
            "coverages":  r2_score(y_train, pred_train),  
            "avg_widths":  mean_absolute_error(y_test, pred_test)}  
    """
    models = {}
    metrics = {}
    pairs = [[quantiles[i], quantiles[-(i+1)]] for i in range(len(quantiles)//2)]
    preds = {}
    for q in quantiles:
        print(f"Training quantile={q}")
        cat_model = CatBoostRegressor(
            iterations=2000,
            learning_rate=0.05,
            depth=8,
            subsample=0.8,
            random_seed=42,
            loss_function=f'Quantile:alpha={q}',
            verbose=200
        )

        train_pool = Pool(X_train, y_train)
        test_pool = Pool(X_test, y_test)
        cat_model.fit(train_pool, eval_set=test_pool, use_best_model=False)

        y_pred = cat_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)

        models[q] = cat_model
        metrics[q] = {'test_MAE': mae}
        preds[q] = y_pred
    
    coverages = []
    avg_widths = [] 
    for low, high in pairs:
        pred_low, pred_high = preds[low], preds[high]
        inside = (y_test >= pred_low) & (y_test <= pred_high)
        coverage = inside.mean() * 100.0
        avg_width = np.mean(pred_high - pred_low)
        coverages.append(coverage)
        avg_widths.append(avg_width)
    
    data = {'preds': preds,
            'coverages': coverages,
            'avg_widths': avg_widths}
    
    return models, data


############################ MLP Models ################################
class MLP(nn.Module):
    """
    Basic Multilayer perceptron network for regression

    Args:
        nn (_type_): _description_
    """
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)
    
def train_mlp_time_series(X_train, y_train, X_test, y_test, model:MLPRegressor):
    """
    Trains a basic mlp regressor

    Args:
        X_train (_type_): _description_
        y_train (_type_): _description_
        X_test (_type_): _description_
        y_test (_type_): _description_
        model (MLPRegressor): _description_

    Returns:
        model (MLPRegressor): trained model  
        metrics (dict): performances  s
            {"train_mae": mean_absolute_error(y_train, pred_train),  
            "train_R2":  r2_score(y_train, pred_train),  
            "MAE":  mean_absolute_error(y_test, pred_test),  
            "R2":   r2_score(y_test, pred_test),  
            "y_pred_test": pred_test,}   
    """
    model = model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred  = model.predict(X_test)
    metrics = {
        "train_mae": mean_absolute_error(y_train, y_train_pred),
        "train_R2":  r2_score(y_train, y_train_pred),
        "MAE":  mean_absolute_error(y_test, y_test_pred),
        "R2":   r2_score(y_test, y_test_pred),
        "y_pred_test": y_test_pred,
    }
    return model, metrics

def train_mlp_quantile(X_train, y_train, X_test, y_test, q, model_param={}):
    """
    Trains a mlp model through quantile regression using a singular quantile value

    Args:
        X_train (NDarray):
        y_train (NDarray): 
        X_test (NDarray):
        y_test (NDarray):
        q (int): selected quantile 
        model (CatBoostRegressor): 

    Returns:
        model (CatBoostRegressor): trained model
        data (dict): performances:
            {'preds': y_preds,
            'trues': y_trues,
            'train_loss': train_losses,
    """
    model = MLP(X_train.shape[1])
    lr = model_param.get('lr', 0.005)
    epochs = model_param.get('epochs', 300)
    batch_size = model_param.get('batch', 1024)
    DEVICE = model_param.get('DEVICE', torch.device("cuda" if torch.cuda.is_available() else "cpu") )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(DEVICE)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test.astype(np.float32)), torch.from_numpy(y_test.astype(np.float32))), batch_size=batch_size, shuffle=False, drop_last=False)
    best_loss = np.inf
    wait = 0
    patience = 5
    train_losses=[]
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train.astype(np.float32)), torch.from_numpy(y_train.astype(np.float32))), batch_size=batch_size, shuffle=True, drop_last=False)
        for xb, yb in train_loader:
            xb = xb.to(DEVICE).float()
            yb = yb.to(DEVICE).float()
            optimizer.zero_grad()
            pred = model(xb).squeeze()
            loss = helpers.quantile_loss(pred, yb, q).to(DEVICE)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_train = total_loss / len(train_loader.dataset)
        train_losses.append(avg_train)  
        print(f"Q={q} Epoch {epoch+1}/{epochs} Loss={avg_train:.4f}")
        # Early stopping
        if avg_train < best_loss - 5e-4:
            best_loss = avg_train
            wait = 0
            best_state = model.state_dict()
        else:
            wait += 1
            if wait >= patience:
                break
        
    model.load_state_dict(best_state)
    y_preds, y_trues = helpers.predict_model(model, test_loader, DEVICE)
    data = {'preds': y_preds,
            'trues': y_trues,
            'train_loss': train_losses,
}
    return model, data


############################ GRU Models ################################
class GRUNetwork(nn.Module):
    """
    Basic GRU network for MTS data regression

    Args:
        input_size (int, optional): The number of features Default is 12.
        hidden_size (int, optional): The number of hidden units in the GRU layer. Default is 256.
        output_size (int, optional): The number of outputs to predict. Default is 1.
        num_layers (int, optional): The number of GRU layers. Default is 3.
        dropout (float, optional): The dropout rate applied to the GRU and fully connected layers. Default is 0.2.  
    """
    def __init__(self, input_size=12, hidden_size=256, output_size=1,
                 num_layers=3, dropout=0.2):
        super(GRUNetwork, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_q = output_size
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        # x: (batch_size, sequence_length, input_size)
        gru_out, _ = self.gru(x)

        # predict last output
        last_output = gru_out[:, -1, :]  # (batch_size, hidden_size)

        output = self.fc_layers(last_output)  # (hidden_size, output_size)

        return output

class PinballLoss(nn.Module):
    """
    Implements the Pinball Loss (Quantile Loss) for quantile regression.
    
    Args:
        quantiles (list[float]): list of quantiles. Must be monotonic ascending
        
    Outputs:
        loss (Tensor): The average Pinball Loss across all quantiles and the batch
    """
    def __init__(self, quantiles=[0.25, 0.75]):
        super(PinballLoss, self).__init__()
        self.quantiles = quantiles

    def forward(self, pred, target):
        # predictions shape: (batch_size, num_quantiles)
        # targets shape: (batch_size, 1)
        losses = []
        for i, q in enumerate(self.quantiles):
            error = target.squeeze() - pred[:, i]
            loss = torch.max((q - 1) * error, q * error)
            losses.append(loss.mean())

        return torch.stack(losses).mean()

def train_gru_time_series(train_ds, test_ds, model_param={}):
    """
    Trains a basic gru regressor
    Args:
        train_ds (TensorDataset): Train Tensor
        test_ds(TensorDataset): Test tensor
        model_param (dict, optional): Model parameters. Defaults to {}.
            Example Parms :
                model_param = {
                    'model': model_gru,
                    'batch_size': 256,
                    'epochs': 100,
                    'objective': objective,
                    'optim': torch.optim.AdamW(model_gru.parameters(), lr=1e-1, weight_decay=1e-4),
                    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
                }
    Returns:
        model: trained model  
        metrics (dict): performances  s
            {"train_mae": mean_absolute_error(y_train, pred_train),  
            "train_R2":  r2_score(y_train, pred_train),  
            "MAE":  mean_absolute_error(y_test, pred_test),  
            "R2":   r2_score(y_test, pred_test),  
            "y_pred_test": pred_test,}     
    """
    model = model_param['model']
    batch_size = model_param['batch_size']
    epochs = model_param['epochs']
    objective = model_param['objective']
    optimizer = model_param['optim']
    
    DEVICE = model_param['device']
    model.to(DEVICE)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    train_losses = []
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)   
        # Mini-batch training
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze(1)
            loss = objective(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += loss.item()

        # Validation
        model.eval()
        avg_train_loss = epoch_loss / (len(train_ds) // batch_size)
        train_losses.append(avg_train_loss)
        print(f" Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f}")
        # Early stopping
        if avg_train_loss < best_val_loss - 5e-4:
            best_val_loss = avg_train_loss
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())# torch.save(model.state_dict(), 'best_gru_rtt_model.pth')
        else:
            patience_counter += 1
            if patience_counter > patience:
                model.load_state_dict(best_state)
                break
            
    y_preds, y_trues = helpers.predict_model(model, test_loader, DEVICE)
    y_preds_train, y_true_trains = helpers.predict_model(model, train_loader, DEVICE)
    metrics = {
        "train_mae": mean_absolute_error(y_true_trains, y_preds_train),
        "train_R2":  r2_score(y_true_trains, y_preds_train),
        "MAE":  mean_absolute_error(y_trues, y_preds),
        "R2":   r2_score(y_trues, y_preds),
        "y_pred_test": y_preds,
    }
    return model, metrics

def train_gru_quantiles(train_ds, test_ds, model_param={}):
    """
    Trains the GRU quantile regression model, using the specified objective within model_param

    Args:
        train_ds (TensorDataset): Train Tensor
        test_ds(TensorDataset): Test tensor
        model_param (dict, optional): Model parameters. Defaults to {}.
            Sample Params:
                    model_param = {
                        'model': model_gru,
                        'batch_size': 256,
                        'epochs': 100,
                        'objective': QuantileLoss(quantiles),
                        'optim': torch.optim.AdamW(model_lstm.parameters(), lr=1e-1, weight_decay=1e-4),
                        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    }
    Returns:
        model: trained model  
        data (dict): preds are matrices of prediction, each column is a prediction for a quantile: 
            data = {'preds': preds_test_mat,
                    'trues': y_test_scaled,
                    'train_loss': train_losses,}
    """
    model = model_param['model']
    batch_size = model_param['batch_size']
    epochs = model_param['epochs']
    objective = model_param['objective']
    optimizer = model_param['optim']
    DEVICE = model_param['device']
    model.to(DEVICE)
    objective.to(DEVICE)

    train_losses = []
    best_state = None
    patience = 5
    wait = 0
    best_val=float('inf')
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    
    print("Begin training: ")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)   
        ####### Training #########################
        for xb, yb in train_loader:
            xb = xb.to(DEVICE).float()
            yb = yb.to(DEVICE).float()
            optimizer.zero_grad()
            preds = model(xb)                  # (B, n_q)
            loss = objective(preds, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_train_loss = total_loss / max(1, len(train_loader.dataset))
        train_losses.append(avg_train_loss)

        model.eval()        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f}")
        if avg_train_loss < best_val - 5e-4:
            best_val = avg_train_loss
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
            'train_loss': train_losses,}
    return model, data


############################ Comparison ################################
def run_basic_models_comparison(df, feature_cols, target_metric, n_lags, sample_size, horizons=[1,2,4,8,16], save_plots=False):
    """
    Runs the basic models comparison between the 4 algorithms. Display comparison results. Optionally store plots in plots/

    Args:
        df (pd.DataFrame): full raw MTS dataset
        feature_cols (list[str]): columns of interest
        target_metric (str): regresion target column name
        n_lags (int): number of previous timestep, not including current timestep t
        sample_size (int): Subset sample size
        horizons (list[int], optional): Defaults to [1,2,4,8,16].
    """
    all_preds = {name:{} for name in ['lstm', 'catboost', 'gru', 'mlp']}
    all_train_mae = {name:[] for name in ['lstm', 'catboost', 'gru', 'mlp']}
    all_test_mae = {name:[] for name in ['lstm', 'catboost', 'gru', 'mlp']}
    all_train_r2 = {name:[] for name in ['lstm', 'catboost', 'gru', 'mlp']}
    all_test_r2 = {name:[] for name in ['lstm', 'catboost', 'gru', 'mlp']}
    if save_plots: path = "../plots/"
    df_numeric = df.select_dtypes(include=[np.number]).copy()
    if 'session_id' in df.columns:
        df_numeric['session_id'] = df['session_id']
    if 'time' in df.columns:
        df_numeric['time'] = df['time']
    # Preslice dataset to prevent leakage
    df_numeric = df_numeric.iloc[:sample_size].copy().reset_index(drop=True)
    assert df_numeric['expt_id'].nunique() == 1
    
    df_numeric[feature_cols] = df_numeric[feature_cols].astype(np.float32)       
    idx = feature_cols.index(target_metric)                 # Index of the regression value, used to inverse transform the scaled value back to original
    train, test = train_test_split(df_numeric, test_size=0.2, shuffle=False)
    scaler = StandardScaler().fit(train.loc[:, feature_cols].astype(np.float32))       # Fit scaler on the important columns
    
    if isinstance(scaler, StandardScaler):
        target_mean = scaler.mean_[idx]                       # Obtain the scaled parameters for rescaling the regression variable
        target_std = scaler.scale_[idx]                        # Each idx holds the parameters of the corresponding idx column
    if isinstance(scaler, MinMaxScaler):
        target_mean = scaler.data_min_[idx]                            # Obtain the scaled parameters for rescaling the regression variable
        target_std = scaler.data_max_[idx]-scaler.data_min_[idx]  # Each idx holds the parameters of the corresponding idx column

    train.loc[:, feature_cols] = scaler.transform(train.loc[:, feature_cols].astype(np.float32))
    test.loc[:, feature_cols] = scaler.transform(test.loc[:,feature_cols].astype(np.float32))

    for i in horizons:
        print(f"Preparing dataset for target='{target_metric}', lags={n_lags}, Horizon={i}")
        train_windows_ds, test_windows_ds = helpers.form_puffer_tensor_dataset(train, test, feature_cols, target_metric, n_lags, horizon=i)
        X_train_lagged, y_train_lagged = helpers.make_lagged_dataset(train.groupby('session_id', sort=False), target_metric, n_lags, horizon=i, feature_cols=feature_cols)
        X_test_lagged, y_test_lagged = helpers.make_lagged_dataset(test.groupby('session_id', sort=False), target_metric, n_lags, horizon=i, feature_cols=feature_cols)
        assert len(test_windows_ds) == len(y_test_lagged)
        
        lstm_model_basic = LSTM_regressor(in_d=6, hidden_d=128, out_d=1, num_layers=4, dropout=0.2)
        basic_model_param = {
            'model': lstm_model_basic,
            'batch_size': 512,
            'epochs': 100,
            'objective': nn.HuberLoss(),
            'optim': torch.optim.AdamW(lstm_model_basic.parameters(), lr=1e-3, weight_decay=0.001),
            'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
        }
        cat_model = CatBoostRegressor(
            iterations=2000, learning_rate=0.05, depth=8,
            subsample=0.8, random_seed=42,
            loss_function="MAE", verbose=200
        )
        
        mlp_model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            learning_rate_init=0.005,
            max_iter=80,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            verbose=True
        )
        
        gru_model_basic = GRUNetwork(input_size=6, hidden_size=128, output_size=1, num_layers=3)
        gru_model_param={
            'model': gru_model_basic,
            'batch_size': 512,
            'epochs': 100,
            'objective': nn.MSELoss(),
            'optim': torch.optim.AdamW(gru_model_basic.parameters(), lr=1e-3, weight_decay=1e-4),
            'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
        }
        
        print(f"Training LSTM")
        lstm_model, lstm_data = train_lstm_timeseries(train_windows_ds, test_windows_ds, basic_model_param)
        all_preds['lstm'][i] = lstm_data.get('y_pred_test', None)
        all_train_mae['lstm'].append(lstm_data.get('train_mae', None))
        all_test_mae['lstm'].append(lstm_data.get('MAE', None))
        all_train_r2['lstm'].append(lstm_data.get('train_R2', None))
        all_test_r2['lstm'].append(lstm_data.get('R2', None))
        
        print(f"Training catboost")
        cat_model, cat_data = train_catboost_time_series(X_train_lagged, y_train_lagged, X_test_lagged, y_test_lagged, cat_model)
        all_preds['catboost'][i] = cat_data.get('y_pred_test', None)
        all_train_mae['catboost'].append(cat_data.get('train_mae', None))
        all_test_mae['catboost'].append(cat_data.get('MAE', None))
        all_train_r2['catboost'].append(cat_data.get('train_R2', None))
        all_test_r2['catboost'].append(cat_data.get('R2', None))
        
        print(f"Training MLP")
        mlp_model, mlp_data = train_mlp_time_series(X_train_lagged, y_train_lagged, X_test_lagged, y_test_lagged, mlp_model)
        all_preds['mlp'][i] = mlp_data.get('y_pred_test', None)
        all_train_mae['mlp'].append(mlp_data.get('train_mae', None))
        all_test_mae['mlp'].append(mlp_data.get('MAE', None))
        all_train_r2['mlp'].append(mlp_data.get('train_R2', None))
        all_test_r2['mlp'].append(mlp_data.get('R2', None))
        
        print(f"Training GRU")
        gru_model, gru_data = train_gru_time_series(train_windows_ds, test_windows_ds, gru_model_param)
        all_preds['gru'][i] = gru_data.get('y_pred_test', None)
        all_train_mae['gru'].append(gru_data.get('train_mae', None))
        all_test_mae['gru'].append(gru_data.get('MAE', None))
        all_train_r2['gru'].append(gru_data.get('train_R2', None))
        all_test_r2['gru'].append(gru_data.get('R2', None))
        
        plt.figure(figsize=(10,4))
        rescaled_trues = y_test_lagged[:200] * target_std + target_mean
        plt.plot(rescaled_trues, label="Actual")
        for (model, c) in [['lstm', 'blue'], ['catboost', 'orange'], ['gru', 'purple'], ['mlp', 'g']]:
            if len(all_preds[model])==0: continue
            rescaled = all_preds[model][i][:200]*target_std + target_mean
            plt.plot(rescaled, '--', color=c, label=f"{model}Pred")
        plt.title(f"{target_metric} {i}-step")
        plt.legend()
        if save_plots: 
            plt.savefig(f'{path}/{target_metric}_horizon_{i}_basic_regression.png')
        plt.tight_layout()
        plt.show()
    
    ########################## Train ###################################    
    plt.figure(figsize=(8,5))
    for (model, m) in [['lstm', 'o'], ['catboost', 's'], ['gru', '^'], ['mlp', 'x']]:
        if all_train_mae[model] == []: continue
        plt.plot(horizons, all_train_mae[model], marker=m, label=f'MAE ({model})')
    plt.xlabel('Horizon')
    plt.ylabel('MAE')
    plt.title(f'Train MAE vs Horizon ({target_metric})')
    plt.xticks(horizons)
    plt.grid(alpha=0.3)
    plt.legend()
    if save_plots: 
        plt.savefig(f'{path}/{target_metric}_train_maes.png')
    plt.tight_layout()
    plt.show()
    
    
    plt.figure(figsize=(8,5))
    for (model, m) in [['lstm', 'o'], ['catboost', 's'], ['gru', '^'], ['mlp', 'x']]:
        if all_train_mae[model] == []: continue
        plt.plot(horizons, all_train_r2[model], marker=m, label=f'R2 ({model})')
    plt.xlabel('Horizon')
    plt.ylabel('MAE')
    plt.title(f'Train R2 vs Horizon ({target_metric})')
    plt.xticks(horizons)
    plt.grid(alpha=0.3)
    plt.legend()
    if save_plots: 
        plt.savefig(f'{path}/{target_metric}_train_r2s.png')
    plt.tight_layout()
    plt.show()
        
    
    ################ Test ######################################    
    plt.figure(figsize=(8,5))
    for (model, m) in [['lstm', 'o'], ['catboost', 's'], ['gru', '^'], ['mlp', 'x']]:
        if all_test_mae[model] == []: continue
        plt.plot(horizons, all_test_mae[model], marker=m, label=f'MAE ({model})')
    plt.xlabel('Horizon')
    plt.ylabel('MAE')
    plt.title(f'Test MAE vs Horizon ({target_metric})')
    plt.xticks(horizons)
    plt.grid(alpha=0.3)
    plt.legend()
    if save_plots: 
        plt.savefig(f'{path}/{target_metric}_test_maes.png')
    plt.tight_layout()
    plt.show()
        
    
    plt.figure(figsize=(8,5))
    for (model, m) in [['lstm', 'o'], ['catboost', 's'], ['gru', '^'], ['mlp', 'x']]:
        if all_test_r2[model] == []: continue
        plt.plot(horizons, all_test_r2[model], marker=m, label=f'R2 ({model})')
    plt.xlabel('Horizon')
    plt.ylabel('R2')
    plt.title(f'Test R2 vs Horizon ({target_metric})')
    plt.xticks(horizons)
    plt.grid(alpha=0.3)
    plt.legend()
    if save_plots: 
        plt.savefig(f'{path}/{target_metric}_test_r2s.png')
    plt.tight_layout()
    plt.show()
        
def run_quantile_models_comparison(df, feature_cols, target_metric, n_lags, sample_size, horizons=[1,2,4,8,16], qs=[0.1, 0.25, 0.75, 0.9], save_plots = False):
    """
    Runs the quantile-prediction models comparison between the 4 algorithms. Display comparison results. Optionally store the plots in plots/

    Args:
        df (pd.DataFrame): full raw MTS dataset
        feature_cols (list[str]): columns of interest
        target_metric (str): regresion target column name
        n_lags (int): number of previous timestep, not including current timestep t
        sample_size (int): Subset sample size
        horizons (list[int], optional): Different horizons. Defaults to [1,2,4,8,16]
        qs (list[float], optional): Different quantile thresholds Defaults to [0.1, 0.25, 0.75, 0.9]
    """
    assert len(qs) % 2 == 0, print('Number of quantiles must be even')
    assert np.all(qs == np.sort(qs)), print('Quantiles must be in monotonic ascending order')
    all_quantile_preds = {name:{} for name in ['lstm', 'catboost', 'gru', 'mlp']}
    all_coverages = {name:{} for name in ['lstm', 'catboost', 'gru', 'mlp']}
    all_bandwidths = {name:{} for name in ['lstm', 'catboost', 'gru', 'mlp']}
    if save_plots: path = "../plots/"
    
    df_numeric = df.select_dtypes(include=[np.number]).copy()
    if 'session_id' in df.columns:
        df_numeric['session_id'] = df['session_id']
    if 'time' in df.columns:
        df_numeric['time'] = df['time']
    # Preslice dataset to prevent leakage
    df_numeric = df_numeric.iloc[:sample_size].copy().reset_index(drop=True)
    assert df_numeric['expt_id'].nunique() == 1
    
    df_numeric[feature_cols] = df_numeric[feature_cols].astype(np.float32)       
    idx = feature_cols.index(target_metric)                 # Index of the regression value, used to inverse transform the scaled value back to original
    train, test = train_test_split(df_numeric, test_size=0.2, shuffle=False)
    scaler = StandardScaler().fit(train.loc[:, feature_cols].astype(np.float32))       # Fit scaler on the important columns
    
    if isinstance(scaler, StandardScaler):
        target_mean = scaler.mean_[idx]                       # Obtain the scaled parameters for rescaling the regression variable
        target_std = scaler.scale_[idx]                        # Each idx holds the parameters of the corresponding idx column
    if isinstance(scaler, MinMaxScaler):
        target_mean = scaler.data_min_[idx]                            # Obtain the scaled parameters for rescaling the regression variable
        target_std = scaler.data_max_[idx]-scaler.data_min_[idx]  # Each idx holds the parameters of the corresponding idx column

    train.loc[:, feature_cols] = scaler.transform(train.loc[:, feature_cols].astype(np.float32))
    test.loc[:, feature_cols] = scaler.transform(test.loc[:,feature_cols].astype(np.float32))
    
    pairs = [[qs[i], qs[-(i+1)]] for i in range(len(qs)//2)]
    labels = [ f'{qs[i]*100}%-{qs[-(i+1)]*100}%' for i in range(len(qs)//2) ]

    for i in horizons:
        print(f"Preparing dataset for target='{target_metric}', lags={n_lags}, Horizon={i}")
        train_windows_ds, test_windows_ds = helpers.form_puffer_tensor_dataset(train, test, feature_cols, target_metric, n_lags, horizon=i)
        X_train_lagged, y_train_lagged = helpers.make_lagged_dataset(train.groupby('session_id', sort=False), target_metric, n_lags, horizon=i, feature_cols=feature_cols)
        X_test_lagged, y_test_lagged = helpers.make_lagged_dataset(test.groupby('session_id', sort=False), target_metric, n_lags, horizon=i, feature_cols=feature_cols)
        assert len(test_windows_ds) == len(y_test_lagged)
        
        print("Training GRU")
        model_gru = GRUNetwork(input_size=6, hidden_size=128, output_size=len(qs), num_layers=2, dropout=0.1)
        model_param = {
            'model': model_gru,
            'batch_size': 256,
            'epochs': 100,
            'objective': PinballLoss(qs),
            'optim': torch.optim.AdamW(model_gru.parameters(), lr=1e-3, weight_decay=1e-4),
            'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
        }
        gru_model, gru_data = train_gru_quantiles(train_windows_ds, test_windows_ds, model_param)
        preds = gru_data['preds']
        trues = gru_data['trues']
        y_true = trues * target_std + target_mean
        all_coverages['gru'][i] = []
        all_bandwidths['gru'][i] = []
        for low, high in pairs:
            low_preds = preds[:, qs.index(low)] * target_std + target_mean
            high_preds = preds[:, qs.index(high)] * target_std + target_mean
            all_quantile_preds['gru'][low] = low_preds
            all_quantile_preds['gru'][high] = high_preds
            coverage, band_width = helpers.visualize_quantiles(low_preds, high_preds, y_true, low, high, target_metric=target_metric, horizon=i, model_name='gru',save_plot=save_plots, 
                                                               path=f'{path}/{target_metric}_horizon{i}_gru_{low*100}_{high*100}_quantreg.png')
            all_coverages['gru'][i].append(coverage)
            all_bandwidths['gru'][i].append(band_width)
        
        
        print("Training LSTM")
        model_lstm = LSTM_regressor(in_d=6, hidden_d=128, out_d=len(qs), num_layers=2, dropout=0.1)
        model_param = {
            'model': model_lstm,
            'batch_size': 256,
            'epochs': 100,
            'objective': QuantileLoss(qs),
            'optim': torch.optim.AdamW(model_lstm.parameters(), lr=1e-3, weight_decay=1e-4),
            'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
        }
        
        lstm_model, lstm_data = train_lstm_quantiles(train_windows_ds, test_windows_ds, model_param)
        preds = lstm_data['preds']
        trues = lstm_data['trues']
        y_true = trues * target_std + target_mean
        all_coverages['lstm'][i] = []
        all_bandwidths['lstm'][i] = []
        for low, high in pairs:
            low_preds = preds[:, qs.index(low)] * target_std + target_mean
            high_preds = preds[:, qs.index(high)] * target_std + target_mean
            all_quantile_preds['lstm'][low] = low_preds
            all_quantile_preds['lstm'][high] = high_preds
            coverage, band_width = helpers.visualize_quantiles(low_preds, high_preds, y_true, low, high, target_metric=target_metric, horizon=i, model_name='LSTM', save_plot=save_plots, 
                                                               path=f'{path}/{target_metric}_horizon{i}_lstm_{low*100}_{high*100}_quantreg.png')
            all_coverages['lstm'][i].append(coverage)
            all_bandwidths['lstm'][i].append(band_width)
        
        print("Training Catboost")
        cat_model, cat_data = train_catboost_quantile(X_train_lagged, y_train_lagged, X_test_lagged, y_test_lagged, quantiles=qs)
        all_quantile_preds['catboost']={j:cat_data['preds'][j] for j in qs}
        all_coverages['catboost'][i] = cat_data['coverages']
        all_bandwidths['catboost'][i] = cat_data['avg_widths']
        y_true = y_test_lagged * target_std + target_mean
        for low, high in pairs:
            low_preds = all_quantile_preds['catboost'][low]* target_std + target_mean
            high_preds = all_quantile_preds['catboost'][high]* target_std + target_mean
            helpers.visualize_quantiles(low_preds, high_preds, y_true, low, high, target_metric=target_metric, horizon=i, model_name="CatBoost", save_plot=save_plots, 
                                                               path=f'{path}/{target_metric}_horizon{i}_catboost_{low*100}_{high*100}_quantreg.png')
                
        print("Training MLP")
        MLP_model_param = {
            'batch_size': 2048,
            'epochs': 100,
            'objective': helpers.quantile_loss,
            'lr': 1e-3,
            'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
        }
        all_coverages['mlp'][i] = []
        all_bandwidths['mlp'][i] = []
        for low, high in pairs:
            MLP_model_low, MLP_data_low = train_mlp_quantile(X_train_lagged, y_train_lagged, X_test_lagged, y_test_lagged, q=low, model_param=MLP_model_param)
            MLP_model_high, MLP_data_high = train_mlp_quantile(X_train_lagged, y_train_lagged, X_test_lagged, y_test_lagged, q=high, model_param=MLP_model_param)
            low_preds = MLP_data_low['preds'] * target_std + target_mean
            high_preds = MLP_data_high['preds'] * target_std + target_mean
            y_true = MLP_data_low['trues'] * target_std + target_mean
            all_quantile_preds['mlp'][low] = low_preds
            all_quantile_preds['mlp'][high] = high_preds
            coverage, band_width = helpers.visualize_quantiles(low_preds, high_preds, y_true, low, high, target_metric=target_metric, horizon=i, model_name='MLP', save_plot=save_plots, 
                                                               path=f'{path}/{target_metric}_horizon{i}_mlp_{low*100}_{high*100}_quantreg.png' )
            all_coverages['mlp'][i].append(coverage)
            all_bandwidths['mlp'][i].append(band_width)
            
        
    ##################### Coverage Comparison Across Horizon between Models #############################
    x = np.arange(len(horizons)) * 1.5
    bar_width = 0.8/4              # 4 model bars per horizon
    fig, axes = plt.subplots(1, len(pairs), figsize=(10, 4))
    axes = np.atleast_1d(axes)
    for i in range(len(labels)):
        ax = axes[i]
        for m_idx, model in enumerate(all_coverages.keys()):
            coverage_dict = all_coverages[model]        # Indiced by horizon
            if not coverage_dict: continue
            coverage = np.asarray([coverage_dict[h] for h in horizons])     # (horizons, num_pairs)
            model_offset = (m_idx - 3 / 2) * bar_width
            xpos = x + model_offset          
            ax.bar(
                xpos,
                coverage[:, i], 
                width=bar_width,
                label=f"{model}"
        )
        ax.set_title(f"{labels[i]} Coverage Across Horizons ({target_metric})")
        ax.set_xlabel('Horizons')
        ax.set_ylabel('Coverage (%)')
        ax.set_xticks(x)  # Place the x-ticks at the horizon positions
        ax.set_xticklabels(horizons)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
    if save_plots: 
        plt.savefig(f'{path}/{target_metric}_coverage_comparison.png')
    plt.tight_layout()
    plt.show()
    
    
    ##################### Interval Width Comparison Across Horizon between Models #############################
    fig, axes = plt.subplots(1, len(pairs), figsize=(10, 4))
    axes = np.atleast_1d(axes)
    
    for i in range(len(labels)):
        if len(pairs) > 1:
            ax = axes[i]
        else:
            ax = axes
        for m_idx, model in enumerate(all_coverages.keys()):
            interval_width = all_bandwidths[model]        # Indiced by horizon
            if not interval_width: continue
            intervals = np.asarray([interval_width[h] for h in horizons])     # (horizons, num_pairs)
            model_offset = (m_idx - 3 / 2) * bar_width
            xpos = x + model_offset          
            ax.bar(
                xpos,
                intervals[:, i], 
                width=bar_width,
                label=f"{model}"
        )
    
        ax.set_title(f"{labels[i]} Average Interval Width Across Horizons ({target_metric})")
        ax.set_xlabel('Horizons')
        ax.set_ylabel('Interval Width')
        ax.set_xticks(x)  # Place the x-ticks at the horizon positions
        ax.set_xticklabels(horizons)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
    if save_plots: 
        plt.savefig(f'{path}/{target_metric}_avg_width_comparison.png')
    plt.tight_layout()
    plt.show()
    