import os
import logging
import warnings
import itertools

import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from dotenv import load_dotenv

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3] if len(Path(__file__).resolve().parents) > 3 else Path(__file__).resolve().parent

load_dotenv(PROJECT_ROOT / ".env")

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1

# Historical window
HISTORY_DAYS = 365

# Target forecast horizon (3 days)
FORECAST_HORIZON = 3

# Final test set length (evaluating rolling 3-day forecasts across these test days)
TEST_DAYS = 30

# ------------------------------------------------------------
# SARIMA tuning search space
# ------------------------------------------------------------

P_VALUES = [0, 1, 2]
D_VALUES = [0, 1]
Q_VALUES = [0, 1, 2]

SEASONAL_P_VALUES = [0, 1]
SEASONAL_D_VALUES = [0, 1]
SEASONAL_Q_VALUES = [0, 1]

SEASONAL_PERIOD = 7

# Step size for rolling validation and test evaluation
VALIDATION_STEP = 1

# Minimum training size during validation
MIN_TRAIN_SIZE = 200


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# LOAD DATA
# ============================================================

def load_training_data() -> pd.Series:

    logger.info("Loading training data from Feature Store...")
    logger.info("Connecting to Hopsworks...")

    try:
        import hopsworks

        project = hopsworks.login(
            api_key_value=os.environ.get("HOPSWORKS_API_KEY", ""),
            project=os.environ.get("HOPSWORKS_PROJECT_NAME", ""),
        )

        fs = project.get_feature_store()

        fg = fs.get_feature_group(
            FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION,
        )

        logger.info("Reading data from Feature Store...")

        df = fg.read()
    except Exception as e:
        logger.warning(f"Could not connect or load from Hopsworks Feature Store: {e}")
        logger.info("Generating synthetic daily AQI series for fallback testing...")
        dates = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=HISTORY_DAYS + 50, freq="D")
        np.random.seed(42)
        trend = np.linspace(100, 150, len(dates))
        seasonality = 30 * np.sin(2 * np.pi * dates.dayofweek / 7)
        noise = np.random.normal(0, 10, len(dates))
        aqi = np.maximum(10, trend + seasonality + noise)
        df = pd.DataFrame({"event_time": dates, "aqi": aqi})

    logger.info(
        f"Loaded {len(df)} raw rows."
    )

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    df["event_time"] = pd.to_datetime(
        df["event_time"],
        utc=True,
    )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    df = df[
        ["event_time", "aqi"]
    ].copy()

    df = df.dropna(
        subset=["event_time", "aqi"]
    )

    # --------------------------------------------------------
    # Daily aggregation
    # --------------------------------------------------------

    logger.info(
        "Preparing daily AQI series..."
    )

    df["date"] = (
        df["event_time"]
        .dt.floor("D")
    )

    daily = (
        df.groupby("date")["aqi"]
        .mean()
        .sort_index()
    )

    logger.info(
        f"Unique daily observations: {len(daily)}"
    )

    logger.info(
        f"Date range: "
        f"{daily.index.min()} → "
        f"{daily.index.max()}"
    )

    # --------------------------------------------------------
    # Continuous blocks
    # --------------------------------------------------------

    dates = daily.index

    date_diff = dates.to_series().diff()

    block_id = (
        date_diff.dt.days
        .ne(1)
        .cumsum()
    )

    blocks = []

    for _, block in daily.groupby(block_id):

        if len(block) > 0:
            blocks.append(block)

    longest_block = max(
        blocks,
        key=len
    )

    logger.info(
        "Longest continuous block:"
    )

    logger.info(
        f"Start: {longest_block.index.min()}"
    )

    logger.info(
        f"End: {longest_block.index.max()}"
    )

    logger.info(
        f"Days: {len(longest_block)}"
    )

    # --------------------------------------------------------
    # Most recent continuous window
    # --------------------------------------------------------

    if len(longest_block) < HISTORY_DAYS:

        raise ValueError(
            f"Not enough continuous data. "
            f"Need {HISTORY_DAYS} days but only "
            f"{len(longest_block)} available."
        )

    series = longest_block.tail(
        HISTORY_DAYS
    )

    series.index = pd.DatetimeIndex(
        series.index,
        freq="D"
    )

    logger.info(
        f"Using latest {HISTORY_DAYS} "
        f"continuous days."
    )

    logger.info(
        f"Selected period: "
        f"{series.index.min()} → "
        f"{series.index.max()}"
    )

    return series


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    actual,
    predicted
):

    actual = np.array(actual)
    predicted = np.array(predicted)

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    mae = mean_absolute_error(
        actual,
        predicted
    )

    r2 = r2_score(
        actual,
        predicted
    )

    return {
        "rmse": round(float(rmse), 3),
        "mae": round(float(mae), 3),
        "r2": round(float(r2), 4),
    }


# ============================================================
# NAIVE BASELINES (3-DAY HORIZON)
# ============================================================

def naive_forecast(
    train,
    horizon=FORECAST_HORIZON
):

    """
    Persistence baseline for target horizon:

        y(t+h) = y(t)
    """

    return np.repeat(
        train.iloc[-1],
        horizon
    )


def seasonal_naive_forecast(
    train,
    horizon=FORECAST_HORIZON,
    seasonal_period=7
):

    """
    Seasonal naive baseline:

        y(t+h) = y(t+h-7)

    Repeats the corresponding weekday from the previous week.
    """

    values = train.iloc[
        -seasonal_period:
    ].values

    repetitions = int(
        np.ceil(
            horizon / seasonal_period
        )
    )

    prediction = np.tile(
        values,
        repetitions
    )

    return prediction[:horizon]


# ============================================================
# CREATE SARIMA MODEL
# ============================================================

def create_sarima(
    series,
    order,
    seasonal_order
):

    model = SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    return model


# ============================================================
# SINGLE VALIDATION FORECAST (ROLLING 3-DAY HORIZON)
# ============================================================

def evaluate_configuration(
    series,
    order,
    seasonal_order,
    horizon=FORECAST_HORIZON,
    step=VALIDATION_STEP,
    min_train_size=MIN_TRAIN_SIZE,
):

    """
    Expanding-window 3-day forecast validation.

    Evaluates predictions specifically on the target 3-day horizon
    using log1p transformed AQI values for numerical stability.
    """

    actuals = []
    predictions = []

    total_length = len(series)

    start = min_train_size

    # Fit initial model on log transformed data
    log_series = np.log1p(series)

    while (
        start + horizon
        <= total_length
    ):

        train_log = log_series.iloc[:start]
        val_actual = series.iloc[start : start + horizon]

        try:

            model = create_sarima(
                train_log,
                order,
                seasonal_order,
            )

            fit = model.fit(
                disp=False,
                maxiter=200,
            )

            forecast_log = fit.forecast(
                steps=horizon
            )

            # Inverse log transform back to original AQI scale
            forecast = np.expm1(forecast_log)

            actuals.extend(val_actual.values)
            predictions.extend(forecast.values)

        except Exception:

            return np.inf

        start += step

    if not actuals:
        return np.inf

    rmse = np.sqrt(
        mean_squared_error(
            actuals,
            predictions,
        )
    )

    return float(rmse)


# ============================================================
# SARIMA TUNING
# ============================================================

def tune_sarima(
    train
):

    logger.info("")
    logger.info("=" * 70)
    logger.info("SARIMA HYPERPARAMETER TUNING (3-DAY ROLLING FORECAST)")
    logger.info("=" * 70)

    candidates = []

    for order in itertools.product(
        P_VALUES,
        D_VALUES,
        Q_VALUES,
    ):

        for seasonal in itertools.product(
            SEASONAL_P_VALUES,
            SEASONAL_D_VALUES,
            SEASONAL_Q_VALUES,
        ):

            seasonal_order = (
                seasonal[0],
                seasonal[1],
                seasonal[2],
                SEASONAL_PERIOD,
            )

            candidates.append(
                (
                    order,
                    seasonal_order,
                )
            )

    logger.info(
        f"Testing {len(candidates)} "
        f"SARIMA configurations..."
    )

    results = []

    for i, (
        order,
        seasonal_order
    ) in enumerate(candidates, 1):

        validation_rmse = (
            evaluate_configuration(
                train,
                order,
                seasonal_order,
                horizon=FORECAST_HORIZON,
                step=VALIDATION_STEP,
                min_train_size=MIN_TRAIN_SIZE,
            )
        )

        results.append({
            "order": order,
            "seasonal_order": seasonal_order,
            "validation_rmse": validation_rmse,
        })

        if i % 10 == 0 or i == len(candidates):
            logger.info(
                f"[{i}/{len(candidates)}] Tested up to "
                f"SARIMA{order}{seasonal_order}"
            )

    # --------------------------------------------------------
    # Sort by validation RMSE
    # --------------------------------------------------------

    results = sorted(
        results,
        key=lambda x: x["validation_rmse"]
    )

    logger.info("")
    logger.info(
        "TOP SARIMA CONFIGURATIONS"
    )

    for result in results[:10]:

        logger.info(
            f"SARIMA"
            f"{result['order']}"
            f"{result['seasonal_order']} "
            f"→ Validation RMSE (3-Day Horizon): "
            f"{result['validation_rmse']:.3f}"
        )

    best = results[0]

    logger.info("")
    logger.info(
        "🏆 BEST SARIMA CONFIGURATION"
    )

    logger.info(
        f"Order: "
        f"{best['order']}"
    )

    logger.info(
        f"Seasonal order: "
        f"{best['seasonal_order']}"
    )

    logger.info(
        f"Validation RMSE: "
        f"{best['validation_rmse']:.3f}"
    )

    return best, results


# ============================================================
# EVALUATE FORECASTING MODELS ON TEST SET (ROLLING 3-DAY HORIZON)
# ============================================================

def evaluate_rolling_test(
    full_series,
    test_days,
    order=None,
    seasonal_order=None,
    model_type="sarima",
    horizon=FORECAST_HORIZON,
    step=VALIDATION_STEP
):

    """
    Evaluates rolling 3-day forecasts across the holdout test period.
    """

    actuals = []
    predictions = []

    total_len = len(full_series)
    train_end = total_len - test_days

    for start in range(train_end, total_len - horizon + 1, step):

        train = full_series.iloc[:start]
        actual = full_series.iloc[start : start + horizon]

        if model_type == "naive":

            pred = naive_forecast(train, horizon=horizon)

        elif model_type == "seasonal_naive":

            pred = seasonal_naive_forecast(train, horizon=horizon, seasonal_period=SEASONAL_PERIOD)

        elif model_type == "sarima":

            # Transform train to log scale
            train_log = np.log1p(train)

            model = create_sarima(
                train_log,
                order,
                seasonal_order
            )

            fit = model.fit(
                disp=False,
                maxiter=500
            )

            forecast_log = fit.forecast(steps=horizon)

            pred = np.expm1(forecast_log).values

        else:

            raise ValueError(f"Unknown model_type: {model_type}")

        actuals.extend(actual.values)
        predictions.extend(pred)

    return calculate_metrics(actuals, predictions)


# ============================================================
# FINAL TRAINING
# ============================================================

def train_and_register():

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    series = load_training_data()

    # --------------------------------------------------------
    # Final chronological test split
    # --------------------------------------------------------

    train = series.iloc[
        :-TEST_DAYS
    ]

    test = series.iloc[
        -TEST_DAYS:
    ]

    logger.info("")
    logger.info("=" * 70)
    logger.info("FINAL TRAIN / TEST SPLIT")
    logger.info("=" * 70)

    logger.info(
        f"Train: {len(train)} rows"
    )

    logger.info(
        f"Test: {len(test)} rows"
    )

    logger.info(
        f"Train period: "
        f"{train.index.min()} → "
        f"{train.index.max()}"
    )

    logger.info(
        f"Test period: "
        f"{test.index.min()} → "
        f"{test.index.max()}"
    )

    # ========================================================
    # BASELINE 1 — NAIVE (ROLLING 3-DAY HORIZON)
    # ========================================================

    logger.info("")
    logger.info(
        "Calculating naive baseline (rolling 3-day horizon)..."
    )

    naive_metrics = evaluate_rolling_test(
        series,
        TEST_DAYS,
        model_type="naive",
        horizon=FORECAST_HORIZON,
        step=VALIDATION_STEP,
    )

    # ========================================================
    # BASELINE 2 — SEASONAL NAIVE (ROLLING 3-DAY HORIZON)
    # ========================================================

    logger.info(
        "Calculating seasonal naive baseline (rolling 3-day horizon)..."
    )

    seasonal_naive_metrics = evaluate_rolling_test(
        series,
        TEST_DAYS,
        model_type="seasonal_naive",
        horizon=FORECAST_HORIZON,
        step=VALIDATION_STEP,
    )

    # ========================================================
    # TUNE SARIMA
    # ========================================================

    best_config, tuning_results = (
        tune_sarima(train)
    )

    best_order = best_config[
        "order"
    ]

    best_seasonal_order = (
        best_config[
            "seasonal_order"
        ]
    )

    # ========================================================
    # FINAL SARIMA EVALUATION ON TEST SET
    # ========================================================

    logger.info("")
    logger.info("=" * 70)
    logger.info("EVALUATING FINAL SARIMA ON 3-DAY ROLLING TEST SET")
    logger.info("=" * 70)

    logger.info(
        f"SARIMA{best_order}"
        f"{best_seasonal_order}"
    )

    sarima_metrics = evaluate_rolling_test(
        series,
        TEST_DAYS,
        order=best_order,
        seasonal_order=best_seasonal_order,
        model_type="sarima",
        horizon=FORECAST_HORIZON,
        step=VALIDATION_STEP,
    )

    # ========================================================
    # FINAL FIT FOR DEPLOYMENT / REGISTRY
    # ========================================================

    train_log = np.log1p(series)
    final_model = create_sarima(
        train_log,
        best_order,
        best_seasonal_order
    )

    fit = final_model.fit(
        disp=False,
        maxiter=500
    )

    logger.info(
        f"Final Model AIC: "
        f"{fit.aic:.2f}"
    )

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    logger.info("")
    logger.info("=" * 70)
    logger.info("FINAL MODEL COMPARISON (3-DAY FORECAST HORIZON)")
    logger.info("=" * 70)

    logger.info(
        f"Naive          → "
        f"RMSE: {naive_metrics['rmse']} | "
        f"MAE: {naive_metrics['mae']} | "
        f"R²: {naive_metrics['r2']}"
    )

    logger.info(
        f"Seasonal Naive → "
        f"RMSE: "
        f"{seasonal_naive_metrics['rmse']} | "
        f"MAE: "
        f"{seasonal_naive_metrics['mae']} | "
        f"R²: "
        f"{seasonal_naive_metrics['r2']}"
    )

    logger.info(
        f"Tuned SARIMA   → "
        f"RMSE: {sarima_metrics['rmse']} | "
        f"MAE: {sarima_metrics['mae']} | "
        f"R²: {sarima_metrics['r2']}"
    )

    logger.info("=" * 70)

    # ========================================================
    # IMPROVEMENT
    # ========================================================

    rmse_improvement = (
        (
            naive_metrics["rmse"]
            - sarima_metrics["rmse"]
        )
        / naive_metrics["rmse"]
    ) * 100

    mae_improvement = (
        (
            naive_metrics["mae"]
            - sarima_metrics["mae"]
        )
        / naive_metrics["mae"]
    ) * 100

    logger.info(
        f"SARIMA RMSE improvement over Naive: "
        f"{rmse_improvement:.2f}%"
    )

    logger.info(
        f"SARIMA MAE improvement over Naive: "
        f"{mae_improvement:.2f}%"
    )

    if (
        sarima_metrics["rmse"]
        < naive_metrics["rmse"]
    ):

        logger.info(
            "✅ Tuned SARIMA beats Naive."
        )

    else:

        logger.warning(
            "⚠️ Tuned SARIMA does NOT beat Naive."
        )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    artifact_path = (
        "/tmp/sarima.pkl"
    )

    joblib.dump(
        fit,
        artifact_path
    )

    logger.info(
        f"Saved SARIMA artifact to "
        f"{artifact_path}"
    )

    # ========================================================
    # REGISTER MODEL
    # ========================================================

    try:
        logger.info(
            "Registering SARIMA model..."
        )

        import hopsworks

        project = hopsworks.login(
            api_key_value=os.environ.get("HOPSWORKS_API_KEY", ""),
            project=os.environ.get("HOPSWORKS_PROJECT_NAME", ""),
        )

        mr = project.get_model_registry()

        model_metrics = {
            "rmse": sarima_metrics["rmse"],
            "mae": sarima_metrics["mae"],
            "r2": sarima_metrics["r2"],
            "validation_rmse": round(
                best_config[
                    "validation_rmse"
                ],
                3
            ),
            "naive_rmse": naive_metrics[
                "rmse"
            ],
            "seasonal_naive_rmse":
                seasonal_naive_metrics[
                    "rmse"
                ],
        }

        sarima_model = (
            mr.sklearn.create_model(
                name="sarima",
                metrics=model_metrics,
                description=(
                    "Tuned SARIMA model for "
                    "daily AQI 3-day forecast. "
                    f"Uses {HISTORY_DAYS}-day "
                    "continuous historical window "
                    f"with log-transformed targets and a {TEST_DAYS}-day "
                    "rolling 3-day holdout evaluation. "
                    "Hyperparameters selected using "
                    "expanding-window 3-day forecast validation. "
                    f"Final configuration: "
                    f"SARIMA{best_order}"
                    f"{best_seasonal_order}."
                ),
            )
        )

        sarima_model.save(
            artifact_path
        )

        logger.info(
            f"Registered SARIMA "
            f"v{sarima_model.version} "
            f"in Hopsworks Model Registry."
        )
    except Exception as e:
        logger.warning(f"Skipped model registration: {e}")

    return {
        "sarima_metrics": sarima_metrics,
        "naive_metrics": naive_metrics,
        "seasonal_naive_metrics":
            seasonal_naive_metrics,
        "best_order": best_order,
        "best_seasonal_order":
            best_seasonal_order,
        "validation_rmse":
            best_config[
                "validation_rmse"
            ],
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train_and_register()
