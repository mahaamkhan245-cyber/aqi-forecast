"""
hopsworks_training_pipeline.py
Pulls features from Hopsworks Feature Store,
trains 3 models (RF, Ridge, PyTorch LSTM),
registers the best in Hopsworks Model Registry.

Run: python hopsworks_training_pipeline.py
"""

import os, sys, warnings, tempfile, shutil
from datetime import datetime

import joblib, numpy as np, pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT_NAME", "pearl_aqi")
MODELS_DIR        = "data/models"
REGISTRY_PATH     = "data/registry/model_registry.csv"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs("data/registry", exist_ok=True)

print("=" * 60)
print("  Pearls AQI — Hopsworks Training Pipeline")
print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 60)

# ── Connect ───────────────────────────────────────────────────────────────
print("\n[1/5] Connecting to Hopsworks …")
if not HOPSWORKS_API_KEY:
    print("ERROR: HOPSWORKS_API_KEY not set"); sys.exit(1)
try:
    import hopsworks
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY,
                              project=HOPSWORKS_PROJECT)
    fs = project.get_feature_store()
    print(f"      ✅ Connected: {project.name}")
except ImportError:
    print("ERROR: pip install hopsworks"); sys.exit(1)
except Exception as e:
    print(f"ERROR: {e}"); sys.exit(1)

# ── Pull features ─────────────────────────────────────────────────────────
print("\n[2/5] Pulling features from Feature Store …")
try:
    fg  = fs.get_feature_group(name="aqi_features", version=1)
    df  = fg.read()
    rename_cols = {
    "event_time": "date",
    "aqi": "AQI",
    "aqi_lag_1": "AQI_lag_1",
    "aqi_lag_2": "AQI_lag_2",
    "aqi_lag_3": "AQI_lag_3",
    "aqi_3day_mean": "AQI_3day_mean",
    "aqi_7day_mean": "AQI_7day_mean",
}
 
    df = df.rename(columns=rename_cols)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    drop_cols = ["row_id","_hoodie_commit_time","_hoodie_record_key",
                 "_hoodie_partition_path","_hoodie_file_name"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df.to_csv("data/processed/final_features.csv", index=False)
    print(f"      ✅ {len(df)} rows from 'aqi_features'")
except Exception as e:
    print(f"      Hopsworks read failed: {e} — using local CSV")
    df = pd.read_csv("data/processed/final_features.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"      Local fallback: {len(df)} rows")

# ── Prepare data ──────────────────────────────────────────────────────────
FCOLS  = [c for c in df.columns if c not in ("date", "AQI")]
X, y   = df[FCOLS], df["AQI"]
split  = int(len(df) * 0.80)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

def score(yt, yp):
    return (round(r2_score(yt,yp),4),
            round(mean_absolute_error(yt,yp),4),
            round(mean_squared_error(yt,yp)**0.5,4))

results = []

# ── Train models ──────────────────────────────────────────────────────────
print("\n[3/5] Training 3 models …")

# 1. Random Forest
print("      [1/3] Random Forest …")
rf = RandomForestRegressor(n_estimators=300, max_depth=15,
                            min_samples_leaf=2, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_r2,rf_mae,rf_rmse = score(y_test, rf.predict(X_test))
joblib.dump(rf, os.path.join(MODELS_DIR, "rf_model.pkl"))
results.append({"algorithm":"Random Forest","r2":rf_r2,"mae":rf_mae,
                "rmse":rf_rmse,"model_file":"rf_model.pkl"})
print(f"            R²={rf_r2:.4f}  MAE={rf_mae:.4f}")

# 2. Ridge Regression
print("      [2/3] Ridge Regression …")
sc_r  = StandardScaler()
ridge = Ridge(alpha=10.0)
ridge.fit(sc_r.fit_transform(X_train), y_train)
rd_r2,rd_mae,rd_rmse = score(y_test, ridge.predict(sc_r.transform(X_test)))
joblib.dump({"model":ridge,"scaler":sc_r}, os.path.join(MODELS_DIR,"ridge_model.pkl"))
results.append({"algorithm":"Ridge Regression","r2":rd_r2,"mae":rd_mae,
                "rmse":rd_rmse,"model_file":"ridge_model.pkl"})
print(f"            R²={rd_r2:.4f}  MAE={rd_mae:.4f}")

# 3. PyTorch LSTM
print("      [3/3] PyTorch LSTM …")
pt_ok = False
try:
    import torch, torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    sc_pt   = StandardScaler()
    n_feats = len(FCOLS)
    X_tr_pt = sc_pt.fit_transform(X_train).astype(np.float32).reshape(-1,1,n_feats)
    X_te_pt = sc_pt.transform(X_test).astype(np.float32).reshape(-1,1,n_feats)
    y_tr_pt = y_train.values.astype(np.float32)

    train_dl = DataLoader(
        TensorDataset(torch.tensor(X_tr_pt),
                      torch.tensor(y_tr_pt).unsqueeze(1)),
        batch_size=32, shuffle=True)

    class AQI_LSTM(nn.Module):
        def __init__(self, inp):
            super().__init__()
            self.l1  = nn.LSTM(inp,128,batch_first=True,bidirectional=True)
            self.d1  = nn.Dropout(0.2)
            self.l2  = nn.LSTM(256,64,batch_first=True,bidirectional=False)
            self.d2  = nn.Dropout(0.2)
            self.fc1 = nn.Linear(64,32)
            self.bn  = nn.BatchNorm1d(32)
            self.fc2 = nn.Linear(32,1)
        def forward(self,x):
            o,_=self.l1(x); o=self.d1(o)
            o,_=self.l2(o); o=self.d2(o[:,-1,:])
            return self.fc2(torch.relu(self.bn(self.fc1(o))))

    DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_pt = AQI_LSTM(n_feats).to(DEVICE)
    opt      = torch.optim.Adam(model_pt.parameters(), lr=1e-3)
    sch      = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=7,factor=0.5)
    crit     = nn.MSELoss()

    val_s    = int(len(X_tr_pt)*0.85)
    Xv       = torch.tensor(X_tr_pt[val_s:]).to(DEVICE)
    yv       = torch.tensor(y_tr_pt[val_s:]).unsqueeze(1).to(DEVICE)
    best_val = float("inf"); best_st = None; no_imp = 0

    for ep in range(1,151):
        model_pt.train()
        for Xb,yb in train_dl:
            Xb,yb=Xb.to(DEVICE),yb.to(DEVICE)
            opt.zero_grad(); loss=crit(model_pt(Xb),yb)
            loss.backward(); opt.step()
        model_pt.eval()
        with torch.no_grad(): vl=crit(model_pt(Xv),yv).item()
        sch.step(vl)
        if vl < best_val:
            best_val=vl; best_st={k:v.cpu().clone() for k,v in model_pt.state_dict().items()}; no_imp=0
        else:
            no_imp+=1
        if no_imp>=15: print(f"            Early stop epoch {ep}"); break

    model_pt.load_state_dict(best_st); model_pt.eval()
    with torch.no_grad():
        pt_preds=model_pt(torch.tensor(X_te_pt).to(DEVICE)).cpu().numpy().flatten()

    pt_r2,pt_mae,pt_rmse = score(y_test, pt_preds)
    torch.save({"model_state":best_st,"input_size":n_feats},
               os.path.join(MODELS_DIR,"pytorch_lstm.pt"))
    joblib.dump(sc_pt, os.path.join(MODELS_DIR,"pytorch_scaler.pkl"))
    results.append({"algorithm":"PyTorch LSTM","r2":pt_r2,"mae":pt_mae,
                    "rmse":pt_rmse,"model_file":"pytorch_lstm.pt"})
    print(f"            R²={pt_r2:.4f}  MAE={pt_mae:.4f}")
    pt_ok = True

except ImportError:
    print("            PyTorch not installed — pip install torch")
except Exception as e:
    print(f"            PyTorch failed: {e}")

# ── Best model ────────────────────────────────────────────────────────────
best = max(results, key=lambda x: x["r2"])
print(f"\n      Leaderboard:")
for r in sorted(results, key=lambda x: x["r2"], reverse=True):
    crown = " ← BEST 🏆" if r["algorithm"]==best["algorithm"] else ""
    print(f"        {r['algorithm']:<30} R²={r['r2']:.4f}{crown}")

# Save production copy
if best["algorithm"] == "Random Forest":
    joblib.dump(rf, os.path.join(MODELS_DIR,"best_model.pkl"))
elif best["algorithm"] == "Ridge Regression":
    joblib.dump({"model":ridge,"scaler":sc_r}, os.path.join(MODELS_DIR,"best_model.pkl"))
elif best["algorithm"] == "PyTorch LSTM" and pt_ok:
    joblib.dump({"type":"pytorch","state":best_st,
                 "scaler":sc_pt,"input_size":n_feats},
                os.path.join(MODELS_DIR,"best_model.pkl"))

# ── Update local registry ─────────────────────────────────────────────────
print("\n[4/5] Updating registry …")
new_rows = []

for i, res in enumerate(results):
    new_rows.append({
        "version": i + 1,
        "algorithm": res["algorithm"],
        "model_file": res["model_file"],
        "r2": res["r2"],
        "mae": res["mae"],
        "rmse": res["rmse"],
        "train_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "status": (
            "Production"
            if res["algorithm"] == best["algorithm"]
            else "Challenger"
        ),
    })
pd.DataFrame(new_rows).to_csv(REGISTRY_PATH, index=False)
print(f"      ✅ Registry saved — Production: {best['algorithm']}")

# ── Register in Hopsworks ─────────────────────────────────────────────────
# ── Register in Hopsworks ─────────────────────────────────────────────────
# ── Register in Hopsworks ─────────────────────────────────────────────────
print("\n[5/5] Registering in Hopsworks Model Registry …")

tmpdir = None

try:
    mr = project.get_model_registry()

    # Temporary directory for files that will be registered
    tmpdir = tempfile.mkdtemp()

    # ---------------------------------------------------------
    # 1. Copy the winning model
    # ---------------------------------------------------------
    src = os.path.join(
        MODELS_DIR,
        best["model_file"]
    )

    if not os.path.exists(src):
        raise FileNotFoundError(
            f"Production model not found: {src}"
        )

    shutil.copy(
        src,
        os.path.join(tmpdir, best["model_file"])
    )

    print(f"      ✅ Model prepared: {best['model_file']}")

    # ---------------------------------------------------------
    # 2. PyTorch LSTM requires its scaler during inference
    # ---------------------------------------------------------
    if best["algorithm"] == "PyTorch LSTM":

        scaler_src = os.path.join(
            MODELS_DIR,
            "pytorch_scaler.pkl"
        )

        if not os.path.exists(scaler_src):
            raise FileNotFoundError(
                f"PyTorch scaler not found: {scaler_src}"
            )

        shutil.copy(
            scaler_src,
            os.path.join(
                tmpdir,
                "pytorch_scaler.pkl"
            )
        )

        print("      ✅ PyTorch scaler prepared: pytorch_scaler.pkl")

    # ---------------------------------------------------------
    # 3. Create Hopsworks model
    # ---------------------------------------------------------
    hw_m = mr.python.create_model(
        name="aqi_predictor",
        metrics={
            "r2": best["r2"],
            "mae": best["mae"],
            "rmse": best["rmse"],
        },
        description=(
            f"AQI Predictor — {best['algorithm']} "
            f"— Defence Phase 7 Karachi"
        ),
    )

    # ---------------------------------------------------------
    # 4. Upload model files to Hopsworks
    # ---------------------------------------------------------
    hw_m.save(tmpdir)

    print(
        f"      ✅ Registered: aqi_predictor "
        f"({best['algorithm']})"
    )

    print("      Registered files:")

    for filename in os.listdir(tmpdir):
        print(f"         • {filename}")

except Exception as e:

    print(
        f"      ⚠️  Hopsworks registry skipped: {e}"
    )

finally:

    # Always clean temporary directory
    if tmpdir and os.path.exists(tmpdir):
        shutil.rmtree(tmpdir)