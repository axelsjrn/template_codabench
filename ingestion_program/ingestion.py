import json
import sys
import time
from pathlib import Path

# Convert any PosixPath in sys.path to strings for Python 3.7 compatibility
# This must be done before any other imports that might use sys.path
sys.path = [str(p) if isinstance(p, Path) else p for p in sys.path]

import pandas as pd


EVAL_SETS = ["test", "private_test"]


def evaluate_model(model, X_test):
    """
    Evaluate model on test data and return predictions as DataFrame.
    For regression, predictions are continuous values.
    """
    y_pred = model.predict(X_test)
    # Ensure predictions are in a DataFrame format (one column)
    # Handle both 1D and 2D arrays
    if y_pred.ndim == 1:
        return pd.DataFrame(y_pred)
    else:
        return pd.DataFrame(y_pred)


def get_train_data(data_dir):
    data_dir = Path(data_dir)
    training_dir = data_dir / "train"
    X_train = pd.read_csv(training_dir / "train_features.csv")
    y_train = pd.read_csv(training_dir / "train_labels.csv")
    # For regression, ensure y_train is a 1D array/series
    # If y_train is a DataFrame with one column, convert to Series
    if isinstance(y_train, pd.DataFrame):
        if y_train.shape[1] == 1:
            y_train = y_train.iloc[:, 0]
        else:
            y_train = y_train.squeeze()
    return X_train, y_train


def main(data_dir, output_dir):
    try:
        # Here, you can import info from the submission module, to evaluate the
        # submission
        # Ensure all paths in sys.path are strings (not PosixPath) for Python 3.7 compatibility
        sys.path = [str(p) for p in sys.path]
        print(f"Importing submission from {sys.path}")
        from submission import get_model
        print("Successfully imported get_model")

        print(f"Loading training data from {data_dir}")
        X_train, y_train = get_train_data(data_dir)
        print(f"Training data loaded: X_train shape {X_train.shape}, y_train shape {y_train.shape}")

        print("Creating model")
        model = get_model()
        print(f"Model created: {type(model)}")

        print("Training the model")
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start
        print(f"Model trained in {train_time:.2f} seconds")
        
        print("-" * 10)
        print("Evaluate the model")
        start = time.time()
        res = {}
        for eval_set in EVAL_SETS:
            test_file = data_dir / eval_set / f"{eval_set}_features.csv"
            print(f"Loading test data from {test_file}")
            if not test_file.exists():
                raise FileNotFoundError(f"Test file not found: {test_file}")
            X_test = pd.read_csv(test_file)
            print(f"Test data loaded: {X_test.shape}")
            print(f"Making predictions for {eval_set}...")
            res[eval_set] = evaluate_model(model, X_test)
            print(f"Predictions made: {res[eval_set].shape}")
        test_time = time.time() - start
        print("-" * 10)
        duration = train_time + test_time
        print(f"Completed Prediction. Total duration: {duration}")
    except Exception as e:
        print(f"ERROR in ingestion: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Write output files
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w+") as f:
        json.dump(dict(train_time=train_time, test_time=test_time), f)
    print(f"Metadata written to {metadata_path}")
    
    # Write predictions
    for eval_set in EVAL_SETS:
        filepath = output_dir / f"{eval_set}_predictions.csv"
        res[eval_set].to_csv(filepath, index=False)
        print(f"Predictions for {eval_set} written to {filepath} ({len(res[eval_set])} predictions)")
        
        # Verify file was created
        if not filepath.exists():
            raise FileNotFoundError(f"Failed to create {filepath}")
    
    print()
    print("Ingestion Program finished. Moving on to scoring")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingestion program for codabench"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/app/input_data",
        help="",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/app/output",
        help="",
    )
    parser.add_argument(
        "--submission-dir",
        type=str,
        default="/app/ingested_program",
        help="",
    )

    args = parser.parse_args()
    # Convert paths to strings to avoid PosixPath issues with Python 3.7
    submission_dir_str = str(args.submission_dir)
    program_dir_str = str(Path(__file__).parent.resolve())
    
    # Ensure we're adding strings, not Path objects
    if submission_dir_str not in sys.path:
        sys.path.append(submission_dir_str)
    if program_dir_str not in sys.path:
        sys.path.append(program_dir_str)
    
    # Final safety check: convert any remaining PosixPath to strings
    sys.path = [str(p) if isinstance(p, Path) else p for p in sys.path]

    main(Path(args.data_dir), Path(args.output_dir))
