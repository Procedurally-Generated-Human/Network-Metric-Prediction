import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class GRUNetwork(nn.Module):
    def __init__(self, input_size=12, hidden_size=256, output_size=1,
                 num_layers=3, dropout=0.2):
        super(GRUNetwork, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

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

class RTTDataset(Dataset):

    def __init__(self, data: pd.DataFrame, targets: pd.DataFrame, sequence_length=25, feature_transform=None, target_transform=None, train=True):

        self.data = data[data.columns].values
        self.sequence_length = sequence_length
        self.targets = targets
        self.train = train

        stride = sequence_length // 2

        self.indices = []
        for i in range(0, len(data) - sequence_length, stride):
            if i + sequence_length  < len(data):
                self.indices.append(i)

        if feature_transform is None:
            self.feature_scaler = StandardScaler()
        else:
            self.feature_scaler = feature_transform
        if target_transform is None:
            self.target_scaler = StandardScaler()
        else:
            self.target_scaler = target_transform

        if train:
            self.scaled_features = self.feature_scaler.transform(data)
            self.scaled_targets = self.target_scaler.transform(targets).flatten()
        else:
            self.scaled_features = self.feature_scaler.transform(data)
            self.scaled_targets = self.target_scaler.transform(targets).flatten()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        start_idx = self.indices[idx]

        x = self.scaled_features[start_idx:start_idx + self.sequence_length]
        y = self.scaled_targets[start_idx + self.sequence_length]

        return torch.FloatTensor(x), torch.FloatTensor([y])

class PinballLoss(nn.Module):
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