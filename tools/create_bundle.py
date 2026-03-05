"""
tools/create_bundle.py — TiVA Time-Machine
Crée bundle.zip à uploader sur codabench.org/competitions/upload/
"""

import os
import zipfile

INCLUDE = [
    "competition.yaml",
    "logo.png",
    "requirements.txt",
    "ingestion_program/ingestion.py",
    "ingestion_program/metadata.yaml",
    "scoring_program/scoring.py",
    "scoring_program/metadata.yaml",
    "solution/submission.py",
    "pages/participate.md",
    "pages/seed.md",
    "pages/timeline.md",
    "pages/terms.md",
]

# Données input accessibles aux participants (y_test.csv EXCLU)
INPUT_FILES = [
    "dev_phase/input_data/X_train.csv",
    "dev_phase/input_data/y_train.csv",
    "dev_phase/input_data/X_test.csv",
    # y_test.csv intentionnellement absent
]

# Labels de référence pour le scoring (non visibles par les participants)
REFERENCE_FILES = [
    "dev_phase/reference_data/labels_public.csv",
    "dev_phase/reference_data/labels_private.csv",
]

OUTPUT_BUNDLE = "bundle.zip"


def create_bundle(root="."):
    root = os.path.abspath(root)
    bundle_path = os.path.join(root, OUTPUT_BUNDLE)

    all_files = INCLUDE + INPUT_FILES + REFERENCE_FILES
    missing = [f for f in all_files if not os.path.exists(os.path.join(root, f))]
    if missing:
        print("⚠️  Fichiers manquants (ignorés) :")
        for f in missing:
            print(f"    - {f}")

    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel_path in all_files:
            full = os.path.join(root, rel_path)
            if os.path.exists(full):
                zf.write(full, rel_path)
                print(f"  ✓ {rel_path}")

    size_mb = os.path.getsize(bundle_path) / 1e6
    print(f"\n✅ Bundle créé : {bundle_path} ({size_mb:.1f} MB)")
    print(f"Upload : https://www.codabench.org/competitions/upload/")


if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    create_bundle(repo_root)
