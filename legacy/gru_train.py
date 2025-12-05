import requests
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from gru_model import *


# Set these then run the script

TARGET = 'delivery_rate'
TARGET_NEXT = TARGET + '_next'
TRAIN_QUANTILE = False
QUANTILES = (0.25, 0.75)
HORIZON = (1, 2, 4, 8, 16)

# Training parameters

WINDOW_SIZE = 25
LAG = 1

HIDDEN_DIM = 128
BATCH_SIZE = 128
NUM_LAYERS = 2
LR = 0.0001

# I don't want to run it long enough to actually trigger the early stopping tbh. It improves past 40
EPOCHS = 40
N_WORKERS = 0
PATIENCE = 20
LR_FACTOR = .5

try:
    df = pd.read_csv("2025-02-10T11_2025-02-11T11_video_sent_2025-02-10T11_2025-02-11T11.csv")
except FileNotFoundError:
    response = requests.get(
        "https://storage.googleapis.com/puffer-data-release/2025-02-10T11_2025-02-11T11/video_sent_2025-02-10T11_2025-02-11T11.csv")
    with open("2025-02-10T11_2025-02-11T11_video_sent_2025-02-10T11_2025-02-11T11.csv", 'wb') as f:
        f.write(response.content)
    df = pd.read_csv("2025-02-10T11_2025-02-11T11_video_sent_2025-02-10T11_2025-02-11T11.csv")

df = df.rename(columns={'time (ns GMT)': 'time'})
df['time'] = (df['time'] - df['time'].min()) / 1e9
groups = df.groupby('session_id')

# MODEL DEFINITION

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tqdm import tqdm

# DATA PROCESSING

print("Preparing numeric dataset.")

df_numeric = df.select_dtypes(include=[np.number]).copy()
if 'session_id' in df.columns:
    df_numeric['session_id'] = df['session_id']
if 'time' in df.columns:
    df_numeric['time'] = df['time']

feature_cols = ['cwnd', 'in_flight', 'min_rtt', 'rtt', 'delivery_rate', 'buffer']
feature_cols = [c for c in feature_cols if c in df_numeric.columns]

n_lags = 1

dfs = []
print(f"Adding lag features (per session, {n_lags} lags)...")
for sid, group in tqdm(df_numeric.groupby('session_id'), total=df_numeric['session_id'].nunique()):
    g = group.sort_values('time').copy()
    for lag in range(1, n_lags + 1):
        for col in feature_cols:
            g[f"{col}_lag{lag}"] = g[col].shift(lag)
    dfs.append(g)

sample_size = 800_000

df_model = pd.concat(dfs, axis=0).dropna().reset_index(drop=True)

if len(df_model) > sample_size:
        df_model = df_model.iloc[:sample_size]

df_model = df_model.iloc[:sample_size]

df_model[TARGET_NEXT] = df_model[TARGET].shift(-1)
df_model = df_model.dropna(subset=[TARGET_NEXT]).reset_index(drop=True)

if 'session_id' in df.columns:
    df_model = df_model.drop(columns=['session_id'])
if 'time' in df.columns:
    df_model = df_model.drop(columns=['time'])


feature_cols = [c for c in df_model.columns if c.startswith(tuple(['cwnd', 'in_flight', 'min_rtt', 'rtt', 'delivery_rate', 'buffer'])) and not c.endswith('_next')]

X = df_model[feature_cols]
y = df_model[TARGET_NEXT].to_frame()



def evaluate_model(model, test_loader, scaler):

    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)

            pred_np = outputs.cpu().numpy()
            actual_np = batch_y.cpu().numpy()

            pred_reshaped = pred_np.reshape(-1, 1)
            actual_reshaped = actual_np.reshape(-1, 1)

            pred_original = scaler.inverse_transform(pred_reshaped).flatten()
            actual_original = scaler.inverse_transform(actual_reshaped).flatten()

            predictions.extend(pred_original)
            actuals.extend(actual_original)

    return np.array(predictions), np.array(actuals)

if '__main__' == __name__:

    # MODEL TRAINING

    horizon_metrics = []

    avg_train_loss = 0
    avg_test_loss = 0

    for horizon in HORIZON:

        df_model[TARGET_NEXT] = df_model[TARGET].shift(-horizon)
        df_model = df_model.dropna(subset=[TARGET_NEXT]).reset_index(drop=True)

        # if 'session_id' in df.columns:
        #     df_model = df_model.drop(columns=['session_id'])
        # if 'time' in df.columns:
        #     df_model = df_model.drop(columns=['time'])

        feature_cols = [c for c in df_model.columns if c.startswith(
            tuple(['cwnd', 'in_flight', 'min_rtt', 'rtt', 'delivery_rate', 'buffer'])) and not c.endswith('_next')]

        X = df_model[feature_cols]
        y = df_model[TARGET_NEXT].to_frame()

        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()

        feature_scaler.fit(X)
        target_scaler.fit(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        train_dataset = RTTDataset(X_train, y_train, sequence_length=WINDOW_SIZE, feature_transform=feature_scaler, target_transform=target_scaler)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=N_WORKERS)

        test_dataset = RTTDataset(X_test, y_test, sequence_length=WINDOW_SIZE, feature_transform=feature_scaler, target_transform=target_scaler)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=N_WORKERS)

        print(f"Training RNN (GRU) {horizon}-step model...")

        if TRAIN_QUANTILE:
            model = GRUNetwork(12, HIDDEN_DIM, 2, NUM_LAYERS).to(device)
            criterion = PinballLoss(QUANTILES)
        else:
            model = GRUNetwork(12, HIDDEN_DIM, 1, NUM_LAYERS).to(device)
            criterion = nn.MSELoss()

        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, patience=PATIENCE, factor=0.5)

        train_losses = []
        test_losses = []

        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0

        # training loop
        for epoch in range(EPOCHS):

            model.train()
            epoch_loss = 0

            # Mini-batch training
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                epoch_loss += loss.item()

            # Validation
            model.eval()
            test_loss = 0

            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    test_outputs = model(batch_X)
                    test_loss += criterion(test_outputs, batch_y).item()

            # Calculate average training loss
            avg_train_loss = epoch_loss / (len(X_train) // BATCH_SIZE)
            avg_test_loss = test_loss / len(test_loader)
            train_losses.append(avg_train_loss)
            test_losses.append(avg_test_loss)

            scheduler.step(avg_test_loss)



            # Early stopping
            if avg_test_loss < best_val_loss:
                best_val_loss = avg_test_loss
                patience_counter = 0
                # torch.save(model.state_dict(), 'best_gru_rtt_model.pth')
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch}/{EPOCHS}], '
                      f'Train Loss: {avg_train_loss:.6f}, '
                      f'Val Loss: {avg_test_loss:.6f}, '
                      f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break

            # model.load_state_dict(torch.load('best_gru_rtt_model.pth'))

        print(f"Trained {horizon}-step model")

        # MODEL EVALUATION

        predictions, actuals = evaluate_model(model, test_loader, test_dataset.target_scaler)

        print(f"actuals: {actuals}")

        if not TRAIN_QUANTILE:
            # Regression training
            # Metrics
            mae = mean_absolute_error(actuals, predictions)
            mse = mean_squared_error(actuals, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(actuals, predictions)
            print(f"Test Metrics:")
            print(f"MAE: {mae:.2f} ms")
            print(f"RMSE: {rmse:.2f} ms")
            print(f"r2: {r2:.4f}")

            # Plot predictions vs actuals
            plt.figure(figsize=(15, 10))

            # Plot first 1800:2000 test samples. Randomly picked future values because it "looks" bad on first 200 for some reason
            plt.subplot(2, 1, 1)
            plt.plot(actuals[1800:2000], label=f'Actual {TARGET}', alpha=0.7)
            plt.plot(predictions[1800:2000], label=f'Predicted {TARGET}', alpha=0.7)
            plt.title(f'Predicted vs Actual {TARGET} (Horizon: {horizon})')
            plt.xlabel('Time')
            plt.ylabel('')
            plt.legend()
            plt.grid(True)

            # Scatter plot
            plt.subplot(2, 1, 2)
            plt.scatter(actuals, predictions, alpha=0.5)
            plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', lw=2)
            plt.xlabel(f'Actual {TARGET}')
            plt.ylabel(f'Predicted {TARGET}')
            plt.title(f'Predicted vs Actual {TARGET} (Horizon: {horizon})')
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            horizon_metrics.append({
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'final_train_loss': avg_train_loss,
                'final_val_loss': avg_test_loss
            })

        else:
            # Quantile training
            print(predictions.shape, actuals.shape)

            if len(QUANTILES) >= 2:
                q_low, q_high = min(QUANTILES), max(QUANTILES)
                # todo: sort this instead
                # pred_low, pred_high = preds[0], preds[1]
                pred_low, pred_high = predictions[0::2], predictions[1::2]

                inside = (actuals >= pred_low) & (actuals <= pred_high)
                coverage = inside.mean() * 100.0  # percentage
                avg_width = np.mean(pred_high - pred_low)

                print(f"Coverage: {coverage:.2f}% | Avg. interval width: {avg_width:.4f}")
                plt.figure(figsize=(10, 5))
                plt.plot(actuals[1800:2000], label='Actual', color='black', linewidth=2)

                q_low, q_high = min(QUANTILES), max(QUANTILES)
                plt.fill_between(
                    np.arange(200),
                    pred_low[1800:2000],
                    pred_high[1800:2000],
                    color='skyblue',
                    alpha=0.4,
                    label=f'{int(q_low * 100)}–{int(q_high * 100)}% range'
                )

                plt.plot(pred_low[1800:2000], linestyle='--', color='blue', alpha=0.7, label=f'q={q_low}')
                plt.plot(pred_high[1800:2000], linestyle='--', color='blue', alpha=0.7, label=f'q={q_high}')

                plt.title(f'Quantile Predictions ({horizon}-Step Horizon)')
                plt.xlabel('Time Steps')
                plt.ylabel(TARGET)
                plt.legend()
                plt.tight_layout()
                plt.show()

                horizon_metrics.append({
                    'coverage': coverage,
                    'avg_width': avg_width,
                })

    if not TRAIN_QUANTILE:
        # R^2 vs Horizon
        r2_metrics = [m['r2'] for m in horizon_metrics]
        plt.figure(figsize=(15, 10))
        plt.xticks(HORIZON)
        plt.plot(HORIZON, r2_metrics, label=f'R2 (val)', alpha=0.7)
        plt.title(f'R2 vs Horizon ({TARGET})')
        plt.xlabel('Horizon')
        plt.ylabel('R2')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # MAE vs Horizon
        mae_metrics = [m['mae'] for m in horizon_metrics]
        plt.figure(figsize=(15, 10))
        plt.xticks(HORIZON)
        plt.plot(HORIZON, mae_metrics, label=f'MAE (val)', alpha=0.7)
        plt.title(f'MAE vs Horizon ({TARGET})')
        plt.xlabel('Horizon')
        plt.ylabel('MAE')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Final Loss vs Horizon
        train_metrics = [m['final_train_loss'] for m in horizon_metrics]
        val_metrics = [m['final_val_loss'] for m in horizon_metrics]
        plt.figure(figsize=(15, 10))
        plt.xticks(HORIZON)
        plt.plot(train_metrics, label=f'train loss (final)', alpha=0.7)
        plt.plot(val_metrics, label=f'val loss (final)', alpha=0.7)
        plt.title(f'MAE vs Horizon ({TARGET})')
        plt.xlabel('Horizon')
        plt.ylabel('MAE')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    else:
        width_metrics = [m['avg_width'] for m in horizon_metrics]
        plt.figure(figsize=(15, 10))
        plt.subplot(1, 2, 1)
        plt.bar([str(i) for i in HORIZON], width_metrics, width=0.3)
        plt.xlabel("Horizon")
        plt.ylabel("Average width")
        plt.title(f"{TARGET} average width vs horizon {QUANTILES}")
        plt.grid(True, axis='y')

        coverage_metrics = [m['coverage'] for m in horizon_metrics]
        plt.subplot(1, 2, 2)
        plt.bar([str(i) for i in HORIZON], coverage_metrics, width=0.3)
        plt.xlabel("Horizon")
        plt.ylabel("Coverage")
        plt.title(f"{TARGET} coverage vs horizon {QUANTILES}")
        plt.grid(True, axis='y')
        plt.legend()
        plt.tight_layout()
        plt.show()