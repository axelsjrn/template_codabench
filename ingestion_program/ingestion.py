"""
Ingestion program — TiVA Time-Machine Data Challenge

Chemins Docker fixes (Codabench) :
  /app/input_data/       -> X_train.csv, y_train.csv, X_test.csv
  /app/output/           -> predictions_public.csv, predictions_private.csv
  /app/ingested_program/ -> submission.py du participant

Les prédictions sont sauvegardées AVEC les clés métier
(Year, Source_Country, Target_Country, Sector_Code) pour permettre
une jointure fiable dans le scoring program.
"""

import os
import importlib.util
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# --- Chemins fixes Codabench ---
DATA_DIR       = "/app/input_data"
OUTPUT_DIR     = "/app/output"
SUBMISSION_DIR = "/app/ingested_program"

# Support exécution locale
if not os.path.exists(DATA_DIR):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",       default="dev_phase/input_data")
    parser.add_argument("--output-dir",     default="ingestion_res")
    parser.add_argument("--submission-dir", default="solution")
    args = parser.parse_args()
    DATA_DIR       = args.data_dir
    OUTPUT_DIR     = args.output_dir
    SUBMISSION_DIR = args.submission_dir

os.makedirs(OUTPUT_DIR, exist_ok=True)

KEYS = ["Year", "Source_Country", "Target_Country", "Sector_Code"]


# ── Feature engineering ────────────────────────────────────────────────────

def fill_missing(df):
    macro_cols = [c for c in df.columns if c.endswith("_Source") or c.endswith("_Target")]
    for col in macro_cols:
        group_key = "Source_Country" if col.endswith("_Source") else "Target_Country"
        df[col] = (df.groupby(group_key)[col]
                     .transform(lambda x: x.interpolate(method="linear").ffill().bfill()))
    return df


def feature_engineering(df):
    df["GDP_ratio"]             = np.log1p(df["GDP_USD_Source"]) - np.log1p(df["GDP_USD_Target"])
    df["Tech_gap"]              = df["HighTech_Export_Source"]   - df["HighTech_Export_Target"]
    df["RnD_gap"]               = df["Research_Spend_Source"]    - df["Research_Spend_Target"]
    df["Unemployment_gap"]      = df["Unemployment_Source"]      - df["Unemployment_Target"]
    df["Trade_openness_mean"]   = (df["Trade_Openness_Source"]   + df["Trade_Openness_Target"]) / 2
    df["Trade_openness_diff"]   = df["Trade_Openness_Source"]    - df["Trade_Openness_Target"]
    df["Market_size_target"]    = np.log1p(df["GDP_USD_Target"]  * df["Population_Target"])
    df["GDP_per_capita_target"] = np.log1p(df["GDP_USD_Target"]  / (df["Population_Target"] + 1))
    df["Macro_stability"]       = -np.abs(df["Inflation_Target"])
    df["Digital_gap"]           = df["Internet_Users_Source"]    - df["Internet_Users_Target"]
    df["Tech_intensity_source"] = df["HighTech_Export_Source"]   * df["Research_Spend_Source"]
    df["FDI_ratio"]             = (np.log1p(df["FDI_Inflow_Source"])
                                   / (np.log1p(df["FDI_Inflow_Target"]) + 1e-6))
    df["Year_norm"]   = (df["Year"] - 2000) / 20
    df["Crisis_2008"] = df["Year"].isin([2008, 2009]).astype(int)
    df = pd.get_dummies(df, columns=["Source_Country", "Sector_Code"], drop_first=False)
    drop_cols = [c for c in ["Sector_Name", "Target_Country", "country_x", "country_y", "Sector_Code_GTN"]
                 if c in df.columns]
    return df.drop(columns=drop_cols)


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Chargement des données...")

    X_train_raw = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"), index_col=0)
    y_train_raw = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"), index_col=0)["TiVA_Value_Target"]
    X_test_raw  = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"),  index_col=0)

    # Sauvegarder les clés métier du test AVANT feature engineering
    keys_pub  = X_test_raw[X_test_raw["Year"].between(2016, 2018)][KEYS].reset_index(drop=True)
    keys_priv = X_test_raw[X_test_raw["Year"].between(2019, 2020)][KEYS].reset_index(drop=True)

    # Ajouter target temporaire pour concat cohérent
    X_train_raw["TiVA_Value_Target"] = y_train_raw.values
    X_test_raw["TiVA_Value_Target"]  = 0  # placeholder

    X_train_raw = fill_missing(X_train_raw)
    X_test_raw  = fill_missing(X_test_raw)

    df_all = pd.concat([X_train_raw, X_test_raw], ignore_index=True)
    df_all = feature_engineering(df_all)

    # Split
    train_mask = df_all["Year"] <= 2015
    pub_mask   = df_all["Year"].between(2016, 2018)
    priv_mask  = df_all["Year"].between(2019, 2020)

    drop_cols  = ["TiVA_Value_Target", "Year"]
    X_train    = df_all[train_mask].drop(columns=drop_cols)
    y_train    = df_all[train_mask]["TiVA_Value_Target"]
    X_pub      = df_all[pub_mask].drop(columns=drop_cols)
    X_priv     = df_all[priv_mask].drop(columns=drop_cols)

    # Normalisation
    scaler     = StandardScaler()
    X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_pub_sc   = pd.DataFrame(scaler.transform(X_pub),   columns=X_pub.columns,   index=X_pub.index)
    X_priv_sc  = pd.DataFrame(scaler.transform(X_priv),  columns=X_priv.columns,  index=X_priv.index)

    print(f"  Train : {X_train_sc.shape} | Public : {X_pub_sc.shape} | Privé : {X_priv_sc.shape}")

    # Charger le modèle du participant
    submission_file = os.path.join(SUBMISSION_DIR, "submission.py")
    if not os.path.exists(submission_file):
        raise FileNotFoundError(f"submission.py introuvable dans {SUBMISSION_DIR}")

    spec   = importlib.util.spec_from_file_location("submission", submission_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "get_model"):
        raise AttributeError("submission.py doit définir une fonction get_model()")

    model = module.get_model()
    print(f"Modèle chargé : {type(model).__name__}")

    print("Entraînement...")
    model.fit(X_train_sc, y_train)

    print("Prédictions...")
    y_pred_pub  = model.predict(X_pub_sc)
    y_pred_priv = model.predict(X_priv_sc)

    # Sauvegarder prédictions AVEC les clés métier pour jointure dans scoring
    keys_pub["TiVA_Value_Target"]  = y_pred_pub
    keys_priv["TiVA_Value_Target"] = y_pred_priv

    keys_pub.to_csv(os.path.join(OUTPUT_DIR,  "predictions_public.csv"),  index=False)
    keys_priv.to_csv(os.path.join(OUTPUT_DIR, "predictions_private.csv"), index=False)

    print(f"  predictions_public.csv  : {len(y_pred_pub)} lignes")
    print(f"  predictions_private.csv : {len(y_pred_priv)} lignes")
