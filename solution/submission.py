"""
Baseline submission for the TiVA Time-Machine Data Challenge.
---------------------------------------------------------------
This file is the reference solution that participants can use as a
starting point. It must expose a `get_model()` function that returns
a scikit-learn compatible estimator (implementing fit / predict).

Participants are free to replace the model, add preprocessing steps
inside a Pipeline, or implement any sklearn-compatible estimator.

Scoring metric (lower is better):
    Score = 0.3 * MAE_public(2016-2018) + 0.7 * MAE_private(2019-2020)
    where MAE is computed on TiVA flows aggregated by (Year, Source_Country).
"""

from sklearn.ensemble import RandomForestRegressor


def get_model():
    """
    Return a scikit-learn compatible model.

    The ingestion program will call:
        model = get_model()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    Data are already preprocessed (feature engineering + StandardScaler)
    by the ingestion program, so no additional preprocessing is required
    here — unless you want to override it with a Pipeline.

    Returns
    -------
    model : sklearn estimator
    """
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,         # use all available cores
    )
    return model


# ---------------------------------------------------------------------------
# Quick local test (optional — not run by Codabench)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=500, n_features=41, noise=0.1, random_state=0)
    model = get_model()
    model.fit(X, y)
    preds = model.predict(X[:5])
    print("Sample predictions:", np.round(preds, 2))
    print("submission.py OK")
