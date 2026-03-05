"""
Scoring program for the TiVA Time-Machine Data Challenge.

Metric: Weighted MAE
    Score = 0.3 * MAE_public (2016-2018) + 0.7 * MAE_private (2019-2020)

The MAE is computed on aggregated TiVA flows per (Year, Source_Country),
not at the row level.
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", default="/app/input/ref")
    parser.add_argument("--prediction-dir", default="/app/input/res")
    parser.add_argument("--output-dir",     default="/app/output")
    return parser.parse_args()


def load_csv(path, name):
    full = os.path.join(path, name)
    if not os.path.exists(full):
        raise FileNotFoundError(f"Missing file: {full}")
    return pd.read_csv(full)


def compute_weighted_mae(pred_pub, true_pub, refs_pub,
                         pred_priv, true_priv, refs_priv):
    """
    Aggregate predictions by (Year, Source_Country) then compute weighted MAE.

    Parameters
    ----------
    pred_pub / pred_priv : array-like of float
        Model predictions for public (2016-2018) and private (2019-2020) sets.
    true_pub / true_priv : array-like of float
        Ground-truth TiVA values.
    refs_pub / refs_priv : pd.DataFrame
        Must contain columns 'Year' and 'Source_Country'.

    Returns
    -------
    score  : float  — final weighted score
    mae_pub: float
    mae_priv: float
    df_pub : pd.DataFrame — aggregated results for public set
    df_priv: pd.DataFrame — aggregated results for private set
    """
    def _aggregate(pred, true, refs):
        df = refs[["Year", "Source_Country"]].copy().reset_index(drop=True)
        df["pred"] = pd.Series(pred).values
        df["true"] = pd.Series(true).values
        return df.groupby(["Year", "Source_Country"])[["pred", "true"]].sum()

    df_pub  = _aggregate(pred_pub,  true_pub,  refs_pub)
    df_priv = _aggregate(pred_priv, true_priv, refs_priv)

    mae_pub  = mean_absolute_error(df_pub["true"],  df_pub["pred"])
    mae_priv = mean_absolute_error(df_priv["true"], df_priv["pred"])
    score    = 0.3 * mae_pub + 0.7 * mae_priv

    return score, mae_pub, mae_priv, df_pub, df_priv


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load predictions produced by ingestion program
    # ------------------------------------------------------------------
    pred_pub  = load_csv(args.prediction_dir, "predictions_public.csv")
    pred_priv = load_csv(args.prediction_dir, "predictions_private.csv")

    # ------------------------------------------------------------------
    # 2. Load ground-truth labels + metadata (Year, Source_Country)
    # ------------------------------------------------------------------
    ref_pub  = load_csv(args.reference_dir, "labels_public.csv")
    ref_priv = load_csv(args.reference_dir, "labels_private.csv")

    # ------------------------------------------------------------------
    # 3. Compute score
    # ------------------------------------------------------------------
    score, mae_pub, mae_priv, df_pub, df_priv = compute_weighted_mae(
        pred_pub["TiVA_Value_Target"].values,
        ref_pub["TiVA_Value_Target"].values,
        ref_pub[["Year", "Source_Country"]],
        pred_priv["TiVA_Value_Target"].values,
        ref_priv["TiVA_Value_Target"].values,
        ref_priv[["Year", "Source_Country"]],
    )

    # ------------------------------------------------------------------
    # 4. Write scores.json (Codabench format)
    # ------------------------------------------------------------------
    scores = {
        "score":     round(score,    2),   # final weighted score (lower = better)
        "mae_public":  round(mae_pub,  2),
        "mae_private": round(mae_priv, 2),
    }

    out_path = os.path.join(args.output_dir, "scores.json")
    with open(out_path, "w") as f:
        json.dump(scores, f, indent=2)

    print("=" * 50)
    print(f"  MAE Public  (2016-2018) : {mae_pub:>12,.2f}")
    print(f"  MAE Private (2019-2020) : {mae_priv:>12,.2f}")
    print(f"  Weighted Score          : {score:>12,.2f}")
    print("=" * 50)
    print(f"Scores written to {out_path}")


if __name__ == "__main__":
    main()
