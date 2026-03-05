"""
Scoring program — TiVA Time-Machine Data Challenge

Chemins Docker fixes (Codabench) :
  /app/input/ref/  -> labels_public.csv, labels_private.csv
  /app/input/res/  -> predictions_public.csv, predictions_private.csv
  /app/output/     -> scores.json

Métrique : Score = 0.3 × MAE_public(2016-2018) + 0.7 × MAE_private(2019-2020)
Le MAE est calculé sur les flux agrégés par (Year, Source_Country).
La jointure se fait sur les clés métier pour éviter tout problème d'ordre.
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

KEYS = ["Year", "Source_Country", "Target_Country"]


def load_csv(directory, filename):
    path = os.path.join(directory, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier manquant : {path}")
    return pd.read_csv(path)


def compute_score(pred_pub, ref_pub, pred_priv, ref_priv):
    def aggregate(pred_df, ref_df):
        # Jointure sur les clés métier — robuste à tout problème d'ordre
        merged = ref_df[KEYS + ["TiVA_Value_Target"]].merge(
            pred_df[KEYS + ["TiVA_Value_Target"]],
            on=KEYS,
            suffixes=("_true", "_pred")
        )
        if len(merged) != len(ref_df):
            raise ValueError(
                f"Jointure incomplète : {len(merged)} lignes matchées sur {len(ref_df)} attendues. "
                "Vérifiez que les clés Year/Source_Country/Target_Country/Sector_Code sont cohérentes."
            )
        return merged.groupby(["Year", "Source_Country"])[
            ["TiVA_Value_Target_true", "TiVA_Value_Target_pred"]].sum()

    df_pub  = aggregate(pred_pub,  ref_pub)
    df_priv = aggregate(pred_priv, ref_priv)

    mae_pub  = mean_absolute_error(df_pub["TiVA_Value_Target_true"],  df_pub["TiVA_Value_Target_pred"])
    mae_priv = mean_absolute_error(df_priv["TiVA_Value_Target_true"], df_priv["TiVA_Value_Target_pred"])
    score    = 0.3 * mae_pub + 0.7 * mae_priv

    return score, mae_pub, mae_priv


if __name__ == "__main__":
    pred_pub  = load_csv(RES_DIR, "predictions_public.csv")
    pred_priv = load_csv(RES_DIR, "predictions_private.csv")
    ref_pub   = load_csv(REF_DIR, "labels_public.csv")
    ref_priv  = load_csv(REF_DIR, "labels_private.csv")

    score, mae_pub, mae_priv = compute_score(pred_pub, ref_pub, pred_priv, ref_priv)

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
