"""
tools/setup_data.py
-------------------
Prepares the data splits for the TiVA Time-Machine Data Challenge.

From the master dataset (Global_Value_Chain_Challenge_Master.xlsx or .csv),
this script produces:
    dev_phase/input_data/
        X_train.csv   — features + target, years 2005-2015
        X_test.csv    — features + target, years 2016-2020  (no leakage: target kept for scoring only)

    dev_phase/reference_data/
        labels_public.csv   — true TiVA + Year + Source_Country for 2016-2018
        labels_private.csv  — true TiVA + Year + Source_Country for 2019-2020

Usage
-----
    python tools/setup_data.py --input path/to/master.csv --output-dir .
"""

import os
import argparse
import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="Path to the master dataset (.csv or .xlsx).")
    parser.add_argument("--output-dir", default=".",
                        help="Root directory where dev_phase/ will be created.")
    return parser.parse_args()


def load_master(path):
    if path.endswith(".xlsx"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def main():
    args = parse_args()

    # Output directories
    input_data_dir = os.path.join(args.output_dir, "dev_phase", "input_data")
    ref_data_dir   = os.path.join(args.output_dir, "dev_phase", "reference_data")
    os.makedirs(input_data_dir, exist_ok=True)
    os.makedirs(ref_data_dir,   exist_ok=True)

    # Load
    print(f"Loading master dataset from: {args.input}")
    df = load_master(args.input)
    print(f"  Shape: {df.shape}")
    print(f"  Years: {df['Year'].min()} – {df['Year'].max()}")
    print(f"  Sources: {sorted(df['Source_Country'].unique())}")

    # ---------------------------------------------------------------
    # Split
    # ---------------------------------------------------------------
    train_mask = df["Year"] <= 2015
    pub_mask   = df["Year"].between(2016, 2018)
    priv_mask  = df["Year"].between(2019, 2020)

    df_train = df[train_mask].copy()
    df_pub   = df[pub_mask].copy()
    df_priv  = df[priv_mask].copy()

    # ---------------------------------------------------------------
    # Input data (features + target for train; features only for test)
    # The ingestion program receives TiVA_Value_Target in X_test too,
    # but it must NOT use it — it is stripped during ingestion.
    # ---------------------------------------------------------------
    df_test = pd.concat([df_pub, df_priv], ignore_index=True)

    df_train.to_csv(os.path.join(input_data_dir, "X_train.csv"), index=False)
    df_test.to_csv( os.path.join(input_data_dir, "X_test.csv"),  index=False)

    # ---------------------------------------------------------------
    # Reference labels (used by scoring program)
    # ---------------------------------------------------------------
    label_cols = ["Year", "Source_Country", "Target_Country",
                  "Sector_Code", "TiVA_Value_Target"]

    df_pub[label_cols].to_csv(
        os.path.join(ref_data_dir, "labels_public.csv"), index=False)
    df_priv[label_cols].to_csv(
        os.path.join(ref_data_dir, "labels_private.csv"), index=False)

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print("\n=== Data splits ===")
    print(f"  Train  (≤2015)    : {len(df_train):,} rows → input_data/X_train.csv")
    print(f"  Public (2016-2018): {len(df_pub):,} rows → input_data/X_test.csv (+ reference)")
    print(f"  Private(2019-2020): {len(df_priv):,} rows → input_data/X_test.csv (+ reference)")
    print(f"\nFiles written to: {args.output_dir}/dev_phase/")


if __name__ == "__main__":
    main()
