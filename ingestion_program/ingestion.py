"""
Ingestion program for the TiVA Time-Machine Data Challenge.

Workflow
--------
1. Load training data (X_train, y_train) and test data (X_pub, X_priv).
2. Import the participant's submission (must expose a `get_model()` function).
3. Fit the model on the training set.
4. Generate predictions on public (2016-2018) and private (2019-2020) test sets.
5. Dump predictions as CSV files for the scoring program.
"""

import os
import sys
import time
import importlib.util
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",       default="/app/input_data",
                        help="Directory containing the challenge data files.")
    parser.add_argument("--output-dir",     default="/app/output",
                        help="Directory where predictions will be written.")
    parser.add_argument("--submission-dir", default="/app/ingested_program",
                        help="Directory containing the participant's submission.py.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Feature engineering (mirrors the baseline notebook)
# ---------------------------------------------------------------------------

def fill_missing(df):
    """Linear interpolation per country group, then forward/backward fill."""
    macro_cols = [c for c in df.columns
                  if c.endswith("_Source") or c.endswith("_Target")]
    for col in macro_cols:
        group_key = "Source_Country" if col.endswith("_Source") else "Target_Country"
        df[col] = (
            df.groupby(group_key)[col]
            .transform(lambda x: x.interpolate(method="linear")
                                   .ffill()
                                   .bfill())
        )
    return df


def feature_engineering(df):
    """Build derived features from bilateral macro indicators."""

    # 1. Economic complementarity ratios
    df["GDP_ratio"]           = np.log1p(df["GDP_USD_Source"]) - np.log1p(df["GDP_USD_Target"])
    df["Tech_gap"]            = df["HighTech_Export_Source"]   - df["HighTech_Export_Target"]
    df["RnD_gap"]             = df["Research_Spend_Source"]    - df["Research_Spend_Target"]
    df["Unemployment_gap"]    = df["Unemployment_Source"]      - df["Unemployment_Target"]
    df["Trade_openness_mean"] = (df["Trade_Openness_Source"]   + df["Trade_Openness_Target"]) / 2
    df["Trade_openness_diff"] = df["Trade_Openness_Source"]    - df["Trade_Openness_Target"]

    # 2. Target market size & attractiveness
    df["Market_size_target"]    = np.log1p(df["GDP_USD_Target"] * df["Population_Target"])
    df["GDP_per_capita_target"] = np.log1p(df["GDP_USD_Target"] / (df["Population_Target"] + 1))
    df["Macro_stability"]       = -np.abs(df["Inflation_Target"])

    # 3. Digital & tech connectivity
    df["Digital_gap"]            = df["Internet_Users_Source"] - df["Internet_Users_Target"]
    df["Tech_intensity_source"]  = df["HighTech_Export_Source"] * df["Research_Spend_Source"]
    df["FDI_ratio"]              = (np.log1p(df["FDI_Inflow_Source"])
                                    / (np.log1p(df["FDI_Inflow_Target"]) + 1e-6))

    # 4. Time features
    df["Year_norm"]   = (df["Year"] - 2000) / 20
    df["Crisis_2008"] = df["Year"].isin([2008, 2009]).astype(int)

    # 5. Categorical encoding
    df = pd.get_dummies(df, columns=["Source_Country", "Sector_Code"], drop_first=False)

    # 6. Drop unused columns
    drop_cols = [c for c in ["Sector_Name", "Target_Country", "country_x", "country_y",
                              "Sector_Code_GTN"]
                 if c in df.columns]
    df = df.drop(columns=drop_cols)

    return df


def prepare_data(data_dir):
    """Load raw data, apply feature engineering and split train/pub/priv."""

    train_path = os.path.join(data_dir, "X_train.csv")
    test_path  = os.path.join(data_dir, "X_test.csv")

    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path)

    # Fill missing values
    df_train = fill_missing(df_train)
    df_test  = fill_missing(df_test)

    # Combine for consistent one-hot encoding
    df_all = pd.concat([df_train, df_test], ignore_index=True)
    df_all = feature_engineering(df_all)

    # Re-split
    train_mask = df_all["Year"] <= 2015
    pub_mask   = df_all["Year"].between(2016, 2018)
    priv_mask  = df_all["Year"].between(2019, 2020)

    drop_cols = ["TiVA_Value_Target", "Year"]

    X_train = df_all[train_mask].drop(columns=drop_cols)
    y_train = df_all[train_mask]["TiVA_Value_Target"]

    X_pub   = df_all[pub_mask].drop(columns=drop_cols)
    y_pub   = df_all[pub_mask]["TiVA_Value_Target"]

    X_priv  = df_all[priv_mask].drop(columns=drop_cols)
    y_priv  = df_all[priv_mask]["TiVA_Value_Target"]

    # Z-score normalisation (fit on train only)
    scaler  = StandardScaler()
    X_train_sc = pd.DataFrame(scaler.fit_transform(X_train),
                               columns=X_train.columns, index=X_train.index)
    X_pub_sc   = pd.DataFrame(scaler.transform(X_pub),
                               columns=X_pub.columns, index=X_pub.index)
    X_priv_sc  = pd.DataFrame(scaler.transform(X_priv),
                               columns=X_priv.columns, index=X_priv.index)

    return X_train_sc, y_train, X_pub_sc, y_pub, X_priv_sc, y_priv


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load & prepare data
    # ------------------------------------------------------------------
    print("Loading and preparing data...")
    X_train, y_train, X_pub, y_pub, X_priv, y_priv = prepare_data(args.data_dir)
    print(f"  Train  : {X_train.shape}")
    print(f"  Public : {X_pub.shape}")
    print(f"  Private: {X_priv.shape}")

    # ------------------------------------------------------------------
    # 2. Load participant's submission
    # ------------------------------------------------------------------
    submission_file = os.path.join(args.submission_dir, "submission.py")
    if not os.path.exists(submission_file):
        raise FileNotFoundError(f"submission.py not found in {args.submission_dir}")

    spec   = importlib.util.spec_from_file_location("submission", submission_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "get_model"):
        raise AttributeError("submission.py must define a `get_model()` function.")

    model = module.get_model()
    print(f"\nModel loaded: {type(model).__name__}")

    # ------------------------------------------------------------------
    # 3. Train
    # ------------------------------------------------------------------
    print("Training model...")
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"  Done in {train_time:.1f}s")

    # ------------------------------------------------------------------
    # 4. Predict
    # ------------------------------------------------------------------
    print("Generating predictions...")
    y_pred_pub  = model.predict(X_pub)
    y_pred_priv = model.predict(X_priv)

    # ------------------------------------------------------------------
    # 5. Save predictions
    # ------------------------------------------------------------------
    pd.DataFrame({"TiVA_Value_Target": y_pred_pub}).to_csv(
        os.path.join(args.output_dir, "predictions_public.csv"), index=False)
    pd.DataFrame({"TiVA_Value_Target": y_pred_priv}).to_csv(
        os.path.join(args.output_dir, "predictions_private.csv"), index=False)

    print(f"\nPredictions saved to {args.output_dir}")
    print(f"  predictions_public.csv  : {len(y_pred_pub)} rows")
    print(f"  predictions_private.csv : {len(y_pred_priv)} rows")

    # Runtime metadata (used by scoring program)
    meta = {"train_time_seconds": round(train_time, 2)}
    import json
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(meta, f)


if __name__ == "__main__":
    main()
