"""Deep Learning models using PyTorch.

Defines:
- PyTorchMLPClassifier: Tabular MLP classifier for tasks T1, T2, T3.
- PyTorchMLPRegressor: Tabular MLP regressor for task T4 (both player Elos).
- PyTorchCNNClassifier: Hybrid CNN-tabular classifier for T3.
- PyTorchCNNRegressor: Hybrid CNN-tabular regressor for T4.

All classes implement the standard scikit-learn estimator interface (fit/predict/predict_proba).
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import chess
from tqdm import tqdm

# Set device
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Deep Learning device: {DEVICE}")

def board_to_tensor(board: chess.Board) -> np.ndarray:
    """Represent chess board as an 8x8x12 tensor (6 piece types * 2 colors)."""
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            # 6 pieces for white, 6 for black
            channel = (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
            row = chess.square_rank(sq)
            col = chess.square_file(sq)
            tensor[channel, row, col] = 1.0
    return tensor

def replay_moves_to_board(moves_text: str, ply_limit: int) -> chess.Board:
    """Replay game move UCI tokens to get board state at a specific ply limit."""
    tokens = str(moves_text).split()
    uci_moves = tokens[1::2] # UCI moves are at odd indices
    
    board = chess.Board()
    for i, uci in enumerate(uci_moves):
        if i >= ply_limit:
            break
        try:
            move = chess.Move.from_uci(uci)
            if move in board.legal_moves:
                board.push(move)
            else:
                break
        except Exception:
            break
    return board

class MLPModel(nn.Module):
    """Simple PyTorch MLP model."""
    def __init__(self, input_dim: int, output_dim: int, hidden_layers: list[int] = [256, 128, 64], dropout: float = 0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class CNNModel(nn.Module):
    """Hybrid CNN + Tabular model for Chess."""
    def __init__(self, tabular_dim: int, output_dim: int, fc_layers: list[int] = [256, 128], dropout: float = 0.3):
        super().__init__()
        # CNN for 8x8x12 board
        self.conv = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten() # 128 * 8 * 8 = 8192
        )
        
        # Concat CNN output (8192) with tabular features
        combined_dim = 8192 + tabular_dim
        
        layers = []
        prev_dim = combined_dim
        for h_dim in fc_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.fc = nn.Sequential(*layers)

    def forward(self, boards, tabular):
        conv_out = self.conv(boards)
        x = torch.cat([conv_out, tabular], dim=1)
        return self.fc(x)

class PyTorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """PyTorch MLP Classifier wrapper for sklearn compat."""
    def __init__(self, hidden_layers=[256, 128, 64], dropout=0.3, lr=0.001, epochs=30, batch_size=256, verbose=0):
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.model = None

    def _prepare_data(self, X):
        if isinstance(X, pd.DataFrame):
            # Select only numeric columns
            X_num = X.select_dtypes(include=[np.number])
        else:
            X_num = X
        return X_num

    def fit(self, X, y):
        X_num = self._prepare_data(X)
        X_imp = self.imputer.fit_transform(X_num)
        X_scaled = self.scaler.fit_transform(X_imp)
        
        y_arr = np.array(y, dtype=np.float32).reshape(-1, 1)
        
        # Build model
        self.model = MLPModel(X_scaled.shape[1], 1, self.hidden_layers, self.dropout).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        dataset = TensorDataset(torch.tensor(X_scaled, dtype=torch.float32), torch.tensor(y_arr, dtype=torch.float32))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                optimizer.zero_grad()
                out = self.model(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch_x)
            if self.verbose > 0 and (epoch + 1) % 5 == 0:
                print(f"MLP Epoch {epoch+1}/{self.epochs} - Loss: {epoch_loss / len(X_scaled):.4f}")
                
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        X_num = self._prepare_data(X)
        X_imp = self.imputer.transform(X_num)
        X_scaled = self.scaler.transform(X_imp)
        
        self.model.eval()
        with torch.no_grad():
            x_t = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)
            logits = self.model(x_t)
            probs = torch.sigmoid(logits).cpu().numpy()
            
        return np.hstack([1.0 - probs, probs])

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)

class PyTorchMLPRegressor(BaseEstimator, RegressorMixin):
    """PyTorch MLP Regressor wrapper for predicting both Elos."""
    def __init__(self, hidden_layers=[256, 128, 64], dropout=0.3, lr=0.001, epochs=30, batch_size=256, verbose=0):
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.model = None

    def _prepare_data(self, X):
        if isinstance(X, pd.DataFrame):
            X_num = X.select_dtypes(include=[np.number])
        else:
            X_num = X
        return X_num

    def fit(self, X, y):
        X_num = self._prepare_data(X)
        X_imp = self.imputer.fit_transform(X_num)
        X_scaled = self.scaler.fit_transform(X_imp)
        
        y_arr = np.array(y, dtype=np.float32) # shape: (N, 2)
        
        # Build model
        self.model = MLPModel(X_scaled.shape[1], 2, self.hidden_layers, self.dropout).to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        dataset = TensorDataset(torch.tensor(X_scaled, dtype=torch.float32), torch.tensor(y_arr, dtype=torch.float32))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                optimizer.zero_grad()
                out = self.model(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch_x)
            if self.verbose > 0 and (epoch + 1) % 5 == 0:
                print(f"MLP Reg Epoch {epoch+1}/{self.epochs} - RMSE: {np.sqrt(epoch_loss / len(X_scaled)):.2f}")
                
        return self

    def predict(self, X):
        X_num = self._prepare_data(X)
        X_imp = self.imputer.transform(X_num)
        X_scaled = self.scaler.transform(X_imp)
        
        self.model.eval()
        with torch.no_grad():
            x_t = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)
            preds = self.model(x_t).cpu().numpy()
        return preds

class PyTorchCNNClassifier(BaseEstimator, ClassifierMixin):
    """PyTorch CNN-Tabular Classifier wrapper."""
    def __init__(self, ply_limit=10, fc_layers=[256, 128], dropout=0.3, lr=0.001, epochs=20, batch_size=256, verbose=0):
        self.ply_limit = ply_limit
        self.fc_layers = fc_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.model = None

    def _extract_tensors_and_tabular(self, X):
        # 1. Tabular features (all numeric)
        X_num = X.select_dtypes(include=[np.number])
        
        # 2. Reconstruct board tensors
        moves_col = "first_10_moves_text" if self.ply_limit == 10 else "first_3_moves_text"
        board_tensors = []
        for moves in X[moves_col]:
            board = replay_moves_to_board(moves, self.ply_limit)
            board_tensors.append(board_to_tensor(board))
            
        return np.array(board_tensors), X_num

    def fit(self, X, y):
        # Ensure it is a DataFrame
        assert isinstance(X, pd.DataFrame), "X must be a pandas DataFrame containing move text"
        
        boards_np, tabular_df = self._extract_tensors_and_tabular(X)
        tabular_imp = self.imputer.fit_transform(tabular_df)
        tabular_scaled = self.scaler.fit_transform(tabular_imp)
        
        y_arr = np.array(y, dtype=np.float32).reshape(-1, 1)
        
        self.model = CNNModel(tabular_scaled.shape[1], 1, self.fc_layers, self.dropout).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        dataset = TensorDataset(
            torch.tensor(boards_np, dtype=torch.float32),
            torch.tensor(tabular_scaled, dtype=torch.float32),
            torch.tensor(y_arr, dtype=torch.float32)
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for b_boards, b_tabular, b_y in dataloader:
                b_boards, b_tabular, b_y = b_boards.to(DEVICE), b_tabular.to(DEVICE), b_y.to(DEVICE)
                optimizer.zero_grad()
                out = self.model(b_boards, b_tabular)
                loss = criterion(out, b_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(b_boards)
            if self.verbose > 0 and (epoch + 1) % 5 == 0:
                print(f"CNN Epoch {epoch+1}/{self.epochs} - Loss: {epoch_loss / len(y_arr):.4f}")
                
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        assert isinstance(X, pd.DataFrame)
        boards_np, tabular_df = self._extract_tensors_and_tabular(X)
        tabular_imp = self.imputer.transform(tabular_df)
        tabular_scaled = self.scaler.transform(tabular_imp)
        
        self.model.eval()
        with torch.no_grad():
            b_t = torch.tensor(boards_np, dtype=torch.float32).to(DEVICE)
            t_t = torch.tensor(tabular_scaled, dtype=torch.float32).to(DEVICE)
            logits = self.model(b_t, t_t)
            probs = torch.sigmoid(logits).cpu().numpy()
            
        return np.hstack([1.0 - probs, probs])

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)

class PyTorchCNNRegressor(BaseEstimator, RegressorMixin):
    """PyTorch CNN-Tabular Regressor wrapper for Elo prediction."""
    def __init__(self, ply_limit=10, fc_layers=[256, 128], dropout=0.3, lr=0.001, epochs=20, batch_size=256, verbose=0):
        self.ply_limit = ply_limit
        self.fc_layers = fc_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.model = None

    def _extract_tensors_and_tabular(self, X):
        # 1. Tabular features (all numeric)
        X_num = X.select_dtypes(include=[np.number])
        
        # 2. Reconstruct board tensors
        moves_col = "first_10_moves_text" if self.ply_limit == 10 else "first_3_moves_text"
        board_tensors = []
        for moves in X[moves_col]:
            board = replay_moves_to_board(moves, self.ply_limit)
            board_tensors.append(board_to_tensor(board))
            
        return np.array(board_tensors), X_num

    def fit(self, X, y):
        assert isinstance(X, pd.DataFrame)
        boards_np, tabular_df = self._extract_tensors_and_tabular(X)
        tabular_imp = self.imputer.fit_transform(tabular_df)
        tabular_scaled = self.scaler.fit_transform(tabular_imp)
        
        y_arr = np.array(y, dtype=np.float32)
        
        self.model = CNNModel(tabular_scaled.shape[1], 2, self.fc_layers, self.dropout).to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        dataset = TensorDataset(
            torch.tensor(boards_np, dtype=torch.float32),
            torch.tensor(tabular_scaled, dtype=torch.float32),
            torch.tensor(y_arr, dtype=torch.float32)
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for b_boards, b_tabular, b_y in dataloader:
                b_boards, b_tabular, b_y = b_boards.to(DEVICE), b_tabular.to(DEVICE), b_y.to(DEVICE)
                optimizer.zero_grad()
                out = self.model(b_boards, b_tabular)
                loss = criterion(out, b_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(b_boards)
            if self.verbose > 0 and (epoch + 1) % 5 == 0:
                print(f"CNN Reg Epoch {epoch+1}/{self.epochs} - RMSE: {np.sqrt(epoch_loss / len(y_arr)):.2f}")
                
        return self

    def predict(self, X):
        assert isinstance(X, pd.DataFrame)
        boards_np, tabular_df = self._extract_tensors_and_tabular(X)
        tabular_imp = self.imputer.transform(tabular_df)
        tabular_scaled = self.scaler.transform(tabular_imp)
        
        self.model.eval()
        with torch.no_grad():
            b_t = torch.tensor(boards_np, dtype=torch.float32).to(DEVICE)
            t_t = torch.tensor(tabular_scaled, dtype=torch.float32).to(DEVICE)
            preds = self.model(b_t, t_t).cpu().numpy()
        return preds
