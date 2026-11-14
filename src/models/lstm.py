"""Modèle LSTM pour la prédiction de la RUL (PyTorch)."""
import copy
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

RUL_SCALE = 125.0  # la cible est divisée par 125 pendant l'entraînement (valeurs entre 0 et 1)


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)


class LSTMRegressor(nn.Module):
    """LSTM empilé + petite tête dense.

    Entrée  : (batch, window, n_features)
    Sortie  : (batch,) RUL normalisée
    Le LSTM lit la fenêtre cycle par cycle ; on garde son état caché au dernier
    cycle, qui résume toute la fenêtre, et on le passe à la tête dense.
    """

    def __init__(self, n_features: int, hidden: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)           # (batch, window, hidden)
        last = out[:, -1, :]            # état au dernier cycle de la fenêtre
        return self.head(last).squeeze(-1)


def _loader(X, y, batch_size, shuffle):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y / RUL_SCALE))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_lstm(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=256,
               lr=1e-3, patience=10, device=None, verbose=True):
    """Entraîne le modèle avec arrêt anticipé sur la perte de validation.

    Retourne le modèle (avec les poids de la meilleure époque) et l'historique.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=patience // 2)
    loss_fn = nn.MSELoss()
    tr, va = _loader(X_train, y_train, batch_size, True), _loader(X_val, y_val, 1024, False)

    best, best_state, wait, hist = np.inf, None, 0, {"train_rmse": [], "val_rmse": []}
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # stabilise l'entraînement
            opt.step()
            tot += loss.item() * len(xb)
        train_rmse = np.sqrt(tot / len(tr.dataset)) * RUL_SCALE

        model.eval()
        with torch.no_grad():
            se = sum(((model(xb.to(device)) - yb.to(device)) ** 2).sum().item() for xb, yb in va)
        val_rmse = np.sqrt(se / len(va.dataset)) * RUL_SCALE
        sched.step(val_rmse)
        hist["train_rmse"].append(train_rmse)
        hist["val_rmse"].append(val_rmse)

        if val_rmse < best - 1e-3:
            best, best_state, wait = val_rmse, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
        if verbose and (ep % 5 == 0 or ep == 1):
            print(f"époque {ep:3d} | RMSE train {train_rmse:6.2f} | RMSE val {val_rmse:6.2f} | {time.time() - t0:5.0f}s")
        if wait >= patience:
            if verbose:
                print(f"Arrêt anticipé à l'époque {ep} (meilleure val RMSE = {best:.2f})")
            break

    model.load_state_dict(best_state)
    return model, hist


@torch.no_grad()
def predict_lstm(model, X, batch_size=2048, device=None):
    """Prédit la RUL (en cycles, ramenée entre 0 et 125)."""
    device = device or next(model.parameters()).device
    model.eval()
    preds = [model(torch.from_numpy(X[i:i + batch_size]).to(device)).cpu().numpy()
             for i in range(0, len(X), batch_size)]
    return np.clip(np.concatenate(preds) * RUL_SCALE, 0, RUL_SCALE)
