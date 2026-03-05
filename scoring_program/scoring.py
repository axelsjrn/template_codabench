"""
Scoring program — TiVA Time-Machine Data Challenge

Chemins Docker fixes (Codabench) :
  /app/input/ref/  → données de référence (labels_public.csv, labels_private.csv)
  /app/input/res/  → prédictions de l'ingestion program
  /app/output/     → scores.json à écrire

Métrique : Score = 0.3 × MAE_public(2016-2018) + 0.7 × MAE_private(2019-2020)
Le MAE est calculé sur les flux agrégés par (Year, Source_Country).
"""

import os
import json
import pandas as pd
from sklearn.metrics import mean_absolute_error

# --- Chemins fixes Codabench ---
REF_DIR = os.path.join("/app/input", "ref")
RES_DIR = os.path.join("/app/input", "res")
OUT_DIR = "/app/output"

# Support exécution locale
if not os.path.exists(REF_DIR):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", default="dev_phase/reference_data")
    parser.add_argument("--prediction-dir", default="ingestion_res")
    parser.add_argument("--output-dir",     default="scoring_res")
    args = parser.parse_args()
    REF_DIR = args.reference_dir
    RES_DIR = args.prediction_dir
    OUT_DIR = args.output_dir

os.makedirs(OUT_DIR, exist_ok=True)


def load_csv(directory, filename):
    path = os.path.join(directory, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier manquant : {path}")
    return pd.read_csv(path)


def compute_score(pred_pub, true_pub, refs_pub, pred_priv, true_priv, refs_priv):
    def aggregate(pred, true, refs):
        df = refs[["Year", "Source_Country"]].copy().reset_index(drop=True)
        df["pred"] = pd.Series(pred).values
        df["true"] = pd.Series(true).values
        return df.groupby(["Year", "Source_Country"])[["pred", "true"]].sum()

    df_pub  = aggregate(pred_pub,  true_pub,  refs_pub)
    df_priv = aggregate(pred_priv, true_priv, refs_priv)

    mae_pub  = mean_absolute_error(df_pub["true"],  df_pub["pred"])
    mae_priv = mean_absolute_error(df_priv["true"], df_priv["pred"])
    score    = 0.3 * mae_pub + 0.7 * mae_priv

    return score, mae_pub, mae_priv


if __name__ == "__main__":
    # Charger prédictions
    pred_pub  = load_csv(RES_DIR, "predictions_public.csv")
    pred_priv = load_csv(RES_DIR, "predictions_private.csv")

    # Charger vérités terrain
    ref_pub  = load_csv(REF_DIR, "labels_public.csv")
    ref_priv = load_csv(REF_DIR, "labels_private.csv")

    # Calculer le score
    score, mae_pub, mae_priv = compute_score(
        pred_pub["TiVA_Value_Target"].values,
        ref_pub["TiVA_Value_Target"].values,
        ref_pub[["Year", "Source_Country"]],
        pred_priv["TiVA_Value_Target"].values,
        ref_priv["TiVA_Value_Target"].values,
        ref_priv[["Year", "Source_Country"]],
    )

    # Écrire scores.json
    scores = {
        "score":       round(score,    2),
        "mae_public":  round(mae_pub,  2),
        "mae_private": round(mae_priv, 2),
    }
    with open(os.path.join(OUT_DIR, "scores.json"), "w") as f:
        json.dump(scores, f, indent=2)

    print(f"MAE Public  : {mae_pub:>12,.2f}")
    print(f"MAE Privé   : {mae_priv:>12,.2f}")
    print(f"Score Final : {score:>12,.2f}")
    print(f"scores.json écrit dans {OUT_DIR}")
