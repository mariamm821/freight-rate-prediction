"""
Spotter Freight Rate Prediction Challenge - final pipeline.

1. Time-based validation: Jan-Sep 2025 -> October 2025.
2. Final training on all labeled development data.
3. Predict all 12,000 rows in validation.csv.
4. Fill data/december_chart_inputs.csv.
5. Run the provided score.py.

CatBoost and scikit-learn are included explicitly in requirements.txt because
the assessment says to choose the modeling approach as you see fit and asks
the repository to contain its dependencies.
"""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
SCORER_RESULTS = ROOT / "scorer_results"

TRAIN_FILE = DATA / "train_test.csv"
VALIDATION_FILE = DATA / "validation.csv"
DECEMBER_FILE = DATA / "december_chart_inputs.csv"

FINAL_PREDICTIONS = ROOT / "validation_predictions.csv"
OCTOBER_PREDICTIONS = RESULTS / "october_predictions.csv"

CATEGORICAL = ["pickup", "delivery", "equipment"]

DROP = [
    "load_id",
    "posted_rate",
    "date",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "market_index",
    "quote_signal",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="raise")

    out["day_of_week"] = out["date"].dt.dayofweek
    out["day_of_month"] = out["date"].dt.day
    out["month"] = out["date"].dt.month
    out["day_of_year"] = out["date"].dt.dayofyear

    # Cyclic calendar features.
    out["doy_sin"] = np.sin(2 * np.pi * out["day_of_year"] / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * out["day_of_year"] / 365.25)
    out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7)

    # Preserve information that weight was missing.
    out["weight_missing"] = out["weight"].isna().astype(int)

    return out


def make_model(iterations: int) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=iterations,
        depth=7,
        learning_rate=0.05,
        loss_function="RMSE",
        l2_leaf_reg=10,
        random_seed=42,
        thread_count=8,
        verbose=False,
        allow_writing_files=False,
    )


def prepare_features(train_df, predict_df):
    train = add_features(train_df)
    predict = add_features(predict_df)

    features = [c for c in train.columns if c not in DROP]
    cat_idx = [features.index(c) for c in CATEGORICAL]

    return train[features], predict[features], cat_idx


def predict_with_ensemble(train_df, predict_df):
    X_train, X_predict, cat_idx = prepare_features(train_df, predict_df)

    y = train_df["posted_rate"].astype(float)
    distance = train_df["distance"].astype(float)
    predict_distance = predict_df["distance"].astype(float)

    if (distance <= 0).any() or (predict_distance <= 0).any():
        raise ValueError("Distance must be positive.")

    # Model 1: CatBoost on log-transformed rate.
    log_model = make_model(425)
    log_model.fit(
        X_train,
        np.log1p(y),
        cat_features=cat_idx,
    )
    log_prediction = np.expm1(log_model.predict(X_predict))

    # Model 2: CatBoost on rate per mile.
    rpm_model = make_model(162)
    rpm_model.fit(
        X_train,
        y / distance,
        cat_features=cat_idx,
    )
    rpm_prediction = rpm_model.predict(X_predict) * predict_distance

    # 2:1 ensemble.
    prediction = (2.0 * log_prediction + rpm_prediction) / 3.0

    return np.maximum(prediction, 0.01)


def run_october_validation(train_df):
    dates = pd.to_datetime(train_df["date"])

    train_part = train_df[dates < "2025-10-01"].copy()
    october = train_df[
        (dates >= "2025-10-01") &
        (dates < "2025-11-01")
    ].copy()

    print("\n=== October validation ===")
    print(f"Training rows (Jan-Sep): {len(train_part):,}")
    print(f"Validation rows (October): {len(october):,}")

    predictions = predict_with_ensemble(train_part, october)
    actual = october["posted_rate"].astype(float).to_numpy()

    mae = mean_absolute_error(actual, predictions)
    rmse = np.sqrt(mean_squared_error(actual, predictions))

    nonzero = actual != 0
    mape = (
        np.mean(
            np.abs(
                (actual[nonzero] - predictions[nonzero])
                / actual[nonzero]
            )
        )
        * 100
    )

    r2 = r2_score(actual, predictions)

    RESULTS.mkdir(parents=True, exist_ok=True)

    result = pd.DataFrame({
        "load_id": october["load_id"].to_numpy(),
        "actual_posted_rate": actual,
        "predicted_rate": predictions,
    })

    result["error"] = result["predicted_rate"] - result["actual_posted_rate"]
    result["absolute_error"] = result["error"].abs()
    result["percentage_error"] = np.where(
        result["actual_posted_rate"] != 0,
        result["absolute_error"]
        / result["actual_posted_rate"]
        * 100,
        np.nan,
    )

    result.to_csv(OCTOBER_PREDICTIONS, index=False)

    (RESULTS / "october_metrics.txt").write_text(
        f"Training period: 2025-01-01 to 2025-09-30\n"
        f"Validation period: 2025-10-01 to 2025-10-31\n"
        f"Training rows: {len(train_part):,}\n"
        f"Validation rows: {len(october):,}\n"
        f"MAE: {mae:.2f}\n"
        f"RMSE: {rmse:.2f}\n"
        f"MAPE: {mape:.2f}%\n"
        f"R2: {r2:.3f}\n"
    )

    print(f"MAE : ${mae:,.2f}")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"R2  : {r2:.3f}")

    return mae, rmse, mape, r2


def make_validation_predictions(train_df, validation_df):
    print("\n=== Final model -> validation.csv ===")
    print(f"Training rows: {len(train_df):,}")
    print(f"Rows to predict: {len(validation_df):,}")

    predictions = predict_with_ensemble(train_df, validation_df)

    submission = pd.DataFrame({
        "load_id": validation_df["load_id"],
        "predicted_rate": predictions,
    })[["load_id", "predicted_rate"]]

    if len(submission) != 12000:
        raise ValueError(
            f"Expected 12,000 predictions, found {len(submission)}."
        )

    if submission["load_id"].isna().any():
        raise ValueError("Missing load_id values.")

    if submission["load_id"].duplicated().any():
        raise ValueError("Duplicate load_id values.")

    if submission["predicted_rate"].isna().any():
        raise ValueError("Missing predictions.")

    if not np.isfinite(submission["predicted_rate"]).all():
        raise ValueError("Non-finite predictions found.")

    if (submission["predicted_rate"] <= 0).any():
        raise ValueError("Non-positive predictions found.")

    submission.to_csv(FINAL_PREDICTIONS, index=False)
    print(f"Saved: {FINAL_PREDICTIONS}")


def make_december_predictions(train):
    df = pd.read_csv(DECEMBER_FILE)

    required = ["pickup", "delivery", "distance", "equipment", "weight", "date"]
    if not all(col in df.columns for col in required):
         raise ValueError(
             f"December input must contain these columns: {required}"
         )
    df = df[required].copy()

    print("\n=== Final model -> December ===")
    print(f"Rows to predict: {len(df):,}")

    predictions = predict_with_ensemble(train, df)

    output = df.copy()
    output["predicted_rate"] = predictions

    if len(output) != 31:
        raise ValueError(
            f"Expected 31 December rows, found {len(output)}."
        )

    if output["predicted_rate"].isna().any():
        raise ValueError("Missing December predictions.")

    if not np.isfinite(output["predicted_rate"]).all():
        raise ValueError("Non-finite December predictions.")

    if (output["predicted_rate"] <= 0).any():
        raise ValueError("Non-positive December predictions.")

    output = output[
        [
            "pickup",
            "delivery",
            "distance",
            "equipment",
            "weight",
            "date",
            "predicted_rate",
        ]
    ]

    output.to_csv(DECEMBER_FILE, index=False)
    print(f"Saved: {DECEMBER_FILE}")


def run_score():
    score_file = ROOT / "score.py"

    if not score_file.exists():
        raise FileNotFoundError(
            "Provided score.py is missing from the project root."
        )

    command = [
        sys.executable,
        str(score_file),
        "--predictions",
        str(FINAL_PREDICTIONS),
        "--december-predictions",
        str(DECEMBER_FILE),
    ]

    print("\n=== Running provided score.py ===")
    subprocess.run(command, cwd=ROOT, check=True)

    chart = SCORER_RESULTS / "candidate_december.png"

    if not chart.exists():
        raise FileNotFoundError(
            f"Required chart was not created: {chart}"
        )

    print(f"Created: {chart}")


def main():
    for required in [
        TRAIN_FILE,
        VALIDATION_FILE,
        DECEMBER_FILE,
        ROOT / "score.py",
    ]:
        if not required.exists():
            raise FileNotFoundError(f"Missing required file: {required}")

    train = pd.read_csv(TRAIN_FILE)
    validation = pd.read_csv(VALIDATION_FILE)

    if "posted_rate" not in train.columns:
        raise ValueError(
            "train_test.csv must contain posted_rate."
        )

    if len(validation) != 12000:
        raise ValueError(
            f"validation.csv must contain 12,000 rows; found {len(validation)}."
        )

    print("=== Spotter Freight Rate Prediction Challenge ===")
    print(f"Labeled rows: {len(train):,}")
    print(f"Validation rows: {len(validation):,}")

    # 1. Validate the model on a future month.
    run_october_validation(train)

    # 2. Train final model on all labeled development data.
    make_validation_predictions(train, validation)

    # 3. Predict the fixed December scenario.
    make_december_predictions(train)

    # 4. Validate outputs and create required chart.
    run_score()

    print("\n=== PIPELINE COMPLETE ===")
    print(f"Validation predictions: {FINAL_PREDICTIONS}")
    print(f"December predictions:   {DECEMBER_FILE}")
    print(
        f"December chart:         "
        f"{SCORER_RESULTS / 'candidate_december.png'}"
    )


if __name__ == "__main__":
    main()
