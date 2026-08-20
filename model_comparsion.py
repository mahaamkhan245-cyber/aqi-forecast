"""
train_model_comparison.py
Trains exactly 3 models:
  1. Random Forest      (sklearn)
  2. Ridge Regression   (sklearn)
  3. PyTorch LSTM       (deep learning)

Cleans up old duplicate model files.
Best model by R² is saved as Production in registry.
All 3 models saved for dashboard comparison.

Run:
    pip install torch
    python train_model_comparison.py
"""

import os
import sys
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────
DATA_PATH     = "data/processed/final_features.csv"
REGISTRY_PATH = "data/registry/model_registry.csv"
MODELS_DIR    = "data/models"
TRAIN_RATIO   = 0.80

os.makedirs(MODELS_DIR,    exist_ok=True)
os.makedirs("data/registry", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────
# STEP 0 — CLEAN UP old duplicate model files
# ─────────────────────────────────────────────────────────────────────────
OLD_FILES = [
    "random_forest_v1.pkl",
    "random_forest_v2.pkl",
    "random_forest_aqi.pkl",
    "nn_model.pkl",
]
print("=" * 65)
print("  Pearls AQI Predictor — Multi-Model Training")
print("  Location: Defence Phase 7, Karachi")
print("=" * 65)
print("\n[0/4] Cleaning up old model files …")
for f in OLD_FILES:
    p = os.path.join(MODELS_DIR, f)
    if os.path.exists(p):
        os.remove(p)
        print(f"      Removed: {f}")
    else:
        print(f"      Not found (skipped): {f}")

# ─────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────
if not os.path.exists(DATA_PATH):
    print(f"\nERROR: {DATA_PATH} not found.")
    print("Run feature_engineering.py first.")
    sys.exit(1)

df    = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df    = df.sort_values("date").reset_index(drop=True)

FCOLS = [c for c in df.columns if c not in ("date", "AQI")]
X     = df[FCOLS]
y     = df["AQI"]

split      = int(len(df) * TRAIN_RATIO)
X_train    = X.iloc[:split];   X_test  = X.iloc[split:]
y_train    = y.iloc[:split];   y_test  = y.iloc[split:]
test_dates = df["date"].iloc[split:].reset_index(drop=True)

print(f"\n  Dataset  : {len(df)} rows  ({df['date'].min().date()} → {df['date'].max().date()})")
print(f"  Train    : {len(X_train)} rows")
print(f"  Test     : {len(X_test)} rows")
print(f"  Features : {len(FCOLS)}")
print(f"  AQI      : mean={y.mean():.1f}  std={y.std():.1f}  "
      f"min={y.min():.1f}  max={y.max():.1f}")


def evaluate(name, y_true, y_pred):
    r2   = r2_score(y_true,  y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true,  y_pred) ** 0.5
    print(f"\n  {'─' * 54}")
    print(f"  {name}")
    print(f"  {'─' * 54}")
    print(f"  R²   = {r2:.4f}")
    print(f"  MAE  = {mae:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    return round(r2, 4), round(mae, 4), round(rmse, 4)


results = []


# ─────────────────────────────────────────────────────────────────────────
# MODEL 1 — RANDOM FOREST
# ─────────────────────────────────────────────────────────────────────────
print("\n\n[1/4]  Random Forest")
print("       Training 300 trees, max_depth=15 …")

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)
rf_pred               = rf.predict(X_test)
rf_r2, rf_mae, rf_rmse = evaluate("Random Forest", y_test, rf_pred)

joblib.dump(rf, os.path.join(MODELS_DIR, "rf_model.pkl"))
print(f"\n  Saved → data/models/rf_model.pkl")

results.append({
    "algorithm":  "Random Forest",
    "r2": rf_r2, "mae": rf_mae, "rmse": rf_rmse,
    "model_file": "rf_model.pkl",
    "preds":      rf_pred,
})


# ─────────────────────────────────────────────────────────────────────────
# MODEL 2 — RIDGE REGRESSION
# ─────────────────────────────────────────────────────────────────────────
print("\n\n[2/4]  Ridge Regression")
print("       StandardScaler + alpha=10 …")

scaler_r = StandardScaler()
X_tr_r   = scaler_r.fit_transform(X_train)
X_te_r   = scaler_r.transform(X_test)

ridge = Ridge(alpha=10.0, max_iter=10000)
ridge.fit(X_tr_r, y_train)
ridge_pred                    = ridge.predict(X_te_r)
ridge_r2, ridge_mae, ridge_rmse = evaluate("Ridge Regression", y_test, ridge_pred)

joblib.dump(
    {"model": ridge, "scaler": scaler_r},
    os.path.join(MODELS_DIR, "ridge_model.pkl"),
)
print(f"\n  Saved → data/models/ridge_model.pkl")

results.append({
    "algorithm":  "Ridge Regression",
    "r2": ridge_r2, "mae": ridge_mae, "rmse": ridge_rmse,
    "model_file": "ridge_model.pkl",
    "preds":      ridge_pred,
})


# ─────────────────────────────────────────────────────────────────────────
# MODEL 3 — PYTORCH LSTM
# ─────────────────────────────────────────────────────────────────────────
print("\n\n[3/4]  PyTorch Bidirectional LSTM")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    print(f"       PyTorch {torch.__version__} detected")
    print("       Training Bidirectional LSTM (128 → 64) …")

    DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS     = 150
    BATCH_SIZE = 32
    LR         = 1e-3
    PATIENCE   = 15

    # Scale
    scaler_pt  = StandardScaler()
    X_tr_pt    = scaler_pt.fit_transform(X_train).astype(np.float32)
    X_te_pt    = scaler_pt.transform(X_test).astype(np.float32)
    y_tr_pt    = y_train.values.astype(np.float32)
    y_te_pt    = y_test.values.astype(np.float32)

    # Reshape → (samples, seq_len=1, features)
    n_feats    = X_tr_pt.shape[1]
    X_tr_3d    = X_tr_pt.reshape(-1, 1, n_feats)
    X_te_3d    = X_te_pt.reshape(-1, 1, n_feats)

    # DataLoader
    train_ds   = TensorDataset(
        torch.tensor(X_tr_3d),
        torch.tensor(y_tr_pt).unsqueeze(1))
    train_dl   = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    # ── Model definition ──────────────────────────────────────────────
    class AQI_LSTM(nn.Module):
        def __init__(self, input_size, hidden1=128, hidden2=64):
            super().__init__()
            self.lstm1 = nn.LSTM(input_size, hidden1,
                                  batch_first=True, bidirectional=True)
            self.drop1 = nn.Dropout(0.2)
            self.lstm2 = nn.LSTM(hidden1 * 2, hidden2,
                                  batch_first=True, bidirectional=False)
            self.drop2 = nn.Dropout(0.2)
            self.fc1   = nn.Linear(hidden2, 32)
            self.bn    = nn.BatchNorm1d(32)
            self.relu  = nn.ReLU()
            self.fc2   = nn.Linear(32, 1)

        def forward(self, x):
            out, _  = self.lstm1(x)
            out     = self.drop1(out)
            out, _  = self.lstm2(out)
            out     = self.drop2(out[:, -1, :])   # last timestep
            out     = self.relu(self.bn(self.fc1(out)))
            return self.fc2(out)

    pt_model   = AQI_LSTM(input_size=n_feats).to(DEVICE)
    criterion  = nn.MSELoss()
    optimizer  = torch.optim.Adam(pt_model.parameters(), lr=LR)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=7, factor=0.5, min_lr=1e-6)

    # ── Training loop ─────────────────────────────────────────────────
    val_split   = int(len(X_tr_3d) * 0.85)
    X_val_t     = torch.tensor(X_tr_3d[val_split:]).to(DEVICE)
    y_val_t     = torch.tensor(y_tr_pt[val_split:]).unsqueeze(1).to(DEVICE)

    best_val    = float("inf")
    best_state  = None
    no_improve  = 0

    for epoch in range(1, EPOCHS + 1):
        pt_model.train()
        for Xb, yb in train_dl:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(pt_model(Xb), yb)
            loss.backward()
            optimizer.step()

        # Validation
        pt_model.eval()
        with torch.no_grad():
            val_loss = criterion(pt_model(X_val_t), y_val_t).item()
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.cpu().clone()
                          for k, v in pt_model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            print(f"       Early stopping at epoch {epoch} "
                  f"(best val_loss={best_val:.4f})")
            break

    if epoch == EPOCHS:
        print(f"       Completed all {EPOCHS} epochs")

    # Restore best weights
    pt_model.load_state_dict(best_state)

    # ── Evaluate ──────────────────────────────────────────────────────
    pt_model.eval()
    with torch.no_grad():
        X_te_t   = torch.tensor(X_te_3d).to(DEVICE)
        pt_preds = pt_model(X_te_t).cpu().numpy().flatten()

    pt_r2, pt_mae, pt_rmse = evaluate(
        "PyTorch Bidirectional LSTM (128→64)", y_test, pt_preds)

    # ── Save ──────────────────────────────────────────────────────────
    torch.save({
        "model_state": best_state,
        "input_size":  n_feats,
        "hidden1":     128,
        "hidden2":     64,
    }, os.path.join(MODELS_DIR, "pytorch_lstm.pt"))
    joblib.dump(scaler_pt, os.path.join(MODELS_DIR, "pytorch_scaler.pkl"))
    print(f"\n  Saved → data/models/pytorch_lstm.pt")
    print(f"  Saved → data/models/pytorch_scaler.pkl")

    results.append({
        "algorithm":  "PyTorch LSTM",
        "r2": pt_r2, "mae": pt_mae, "rmse": pt_rmse,
        "model_file": "pytorch_lstm.pt",
        "preds":      pt_preds,
    })

except ImportError:
    print("\n  PyTorch not installed.")
    print("  Run: pip install torch")
    print("  Skipping PyTorch LSTM — only RF and Ridge will be compared.")

except Exception as e:
    print(f"\n  PyTorch LSTM failed: {e}")
    print("  Continuing with RF and Ridge only.")


# ─────────────────────────────────────────────────────────────────────────
# LEADERBOARD
# ─────────────────────────────────────────────────────────────────────────
if not results:
    print("\nERROR: No models trained successfully.")
    sys.exit(1)

best = max(results, key=lambda x: x["r2"])

print("\n\n" + "=" * 65)
print("  LEADERBOARD  (chronological 80/20 test split)")
print("=" * 65)
for r in sorted(results, key=lambda x: x["r2"], reverse=True):
    crown = "  ← BEST 🏆" if r["algorithm"] == best["algorithm"] else ""
    print(
        f"  {r['algorithm']:<36}"
        f"  R²={r['r2']:.4f}"
        f"  MAE={r['mae']:.4f}"
        f"  RMSE={r['rmse']:.4f}"
        f"{crown}"
    )
print("=" * 65)
print(f"\n  🏆  BEST: {best['algorithm']}")
print(f"       R²={best['r2']:.4f}  MAE={best['mae']:.4f}  "
      f"RMSE={best['rmse']:.4f}")


# ─────────────────────────────────────────────────────────────────────────
# SAVE BEST MODEL → production file used by forecast + dashboard
# ─────────────────────────────────────────────────────────────────────────
print("\n\n[4/4]  Saving production model …")

PROD_PATH = os.path.join(MODELS_DIR, "best_model.pkl")

if best["algorithm"] == "Random Forest":
    joblib.dump(rf, PROD_PATH)

elif best["algorithm"] == "Ridge Regression":
    joblib.dump({"model": ridge, "scaler": scaler_r}, PROD_PATH)

elif best["algorithm"] == "PyTorch LSTM":
    # Save a callable wrapper so forecast_prediction.py works transparently
    class PyTorchWrapper:
        """Sklearn-style wrapper around the trained PyTorch LSTM."""
        def __init__(self, state_dict, scaler, input_size,
                     hidden1=128, hidden2=64):
            self.state_dict  = state_dict
            self.scaler      = scaler
            self.input_size  = input_size
            self.hidden1     = hidden1
            self.hidden2     = hidden2

        def _build(self):
            import torch
            import torch.nn as nn

            class _LSTM(nn.Module):
                def __init__(self, inp, h1, h2):
                    super().__init__()
                    self.l1  = nn.LSTM(inp, h1, batch_first=True,
                                        bidirectional=True)
                    self.d1  = nn.Dropout(0.2)
                    self.l2  = nn.LSTM(h1*2, h2, batch_first=True,
                                        bidirectional=False)
                    self.d2  = nn.Dropout(0.2)
                    self.fc1 = nn.Linear(h2, 32)
                    self.bn  = nn.BatchNorm1d(32)
                    self.act = nn.ReLU()
                    self.fc2 = nn.Linear(32, 1)

                def forward(self, x):
                    o,_ = self.l1(x); o = self.d1(o)
                    o,_ = self.l2(o); o = self.d2(o[:,-1,:])
                    return self.fc2(self.act(self.bn(self.fc1(o))))

            m = _LSTM(self.input_size, self.hidden1, self.hidden2)
            m.load_state_dict(self.state_dict)
            m.eval()
            return m

        def predict(self, X):
            import torch
            Xs = self.scaler.transform(X).astype(np.float32)
            Xs = Xs.reshape(-1, 1, self.input_size)
            m  = self._build()
            with torch.no_grad():
                p = m(torch.tensor(Xs)).numpy().flatten()
            return p

    wrapper = PyTorchWrapper(
        state_dict=best_state,
        scaler=scaler_pt,
        input_size=n_feats,
    )
    joblib.dump(wrapper, PROD_PATH)

print(f"  Saved → data/models/best_model.pkl")
print(f"  Algorithm: {best['algorithm']}")


# ─────────────────────────────────────────────────────────────────────────
# UPDATE MODEL REGISTRY  (fresh — no old RF versions)
# ─────────────────────────────────────────────────────────────────────────
new_rows = []
for i, res in enumerate(results):
    new_rows.append({
        "version":    i + 1,
        "algorithm":  res["algorithm"],
        "model_file": res["model_file"],
        "r2":         res["r2"],
        "mae":        res["mae"],
        "rmse":       res["rmse"],
        "train_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "status":     "Production"
                      if res["algorithm"] == best["algorithm"]
                      else "Challenger",
    })

registry = pd.DataFrame(new_rows)
registry.to_csv(REGISTRY_PATH, index=False)

print("\n  Registry (clean):")
print(registry[["version","algorithm","r2","mae","status"]].to_string(index=False))
print(f"\n✅  Training complete!")
print(f"    Best model : {best['algorithm']}  (R²={best['r2']:.4f})")
print(f"    Production : data/models/best_model.pkl")
print(f"    Run        : streamlit run app.py")