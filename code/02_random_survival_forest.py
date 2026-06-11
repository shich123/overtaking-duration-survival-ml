"""Fit and evaluate a Random Survival Forest model for overtaking duration."""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, ParameterGrid
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
MODEL_DIR = ROOT / "outputs" / "models"

TIME_COL = "T"
EVENT_COL = "overtaking"
CATEGORICAL_FEATURES = ["INVASION", "overtaken vehicles", "overtaking vehicles"]
RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    path = TABLE_DIR / "prepared_modeling_dataset.csv"
    if not path.exists():
        raise FileNotFoundError("Run code/00_prepare_data.py before fitting models.")
    return pd.read_csv(path)


def make_design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    x = df.drop(columns=[TIME_COL, EVENT_COL]).copy()
    x = pd.get_dummies(x, columns=CATEGORICAL_FEATURES, drop_first=False, dtype=float)
    return x


def make_survival_target(df: pd.DataFrame) -> np.ndarray:
    return np.array(
        list(zip(df[EVENT_COL].astype(bool), df[TIME_COL].astype(float))),
        dtype=[("event", "?"), ("time", "<f8")],
    )


def c_index(y_true: np.ndarray, risk_scores: np.ndarray) -> float:
    return float(concordance_index_censored(y_true["event"], y_true["time"], risk_scores)[0])


def cross_validate_grid(x: pd.DataFrame, y: np.ndarray) -> tuple[dict, pd.DataFrame]:
    """Small-sample grid search using 5-fold cross-validation.

    The grid is deliberately moderate because the dataset is small. The final
    manuscript should describe the selected ranges if these results are used.
    """
    grid = {
        "n_estimators": [100, 300, 500],
        "min_samples_split": [6, 10],
        "min_samples_leaf": [3, 5, 10],
        "max_features": ["sqrt", 0.5, None],
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    best_score = -np.inf
    best_params = None

    for params in ParameterGrid(grid):
        fold_scores = []
        for train_idx, test_idx in cv.split(x):
            model = RandomSurvivalForest(
                **params,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
            model.fit(x.iloc[train_idx], y[train_idx])
            risk = model.predict(x.iloc[test_idx])
            fold_scores.append(c_index(y[test_idx], risk))

        mean_score = float(np.mean(fold_scores))
        sd_score = float(np.std(fold_scores, ddof=1))
        row = dict(params)
        row.update({"mean_c_index": mean_score, "sd_c_index": sd_score})
        rows.append(row)

        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    return best_params, pd.DataFrame(rows).sort_values("mean_c_index", ascending=False)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    x = make_design_matrix(df)
    y = make_survival_target(df)

    best_params, results = cross_validate_grid(x, y)
    results.to_csv(TABLE_DIR / "rsf_grid_search_results.csv", index=False, encoding="utf-8-sig")

    final_model = RandomSurvivalForest(
        **best_params,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    final_model.fit(x, y)
    apparent_c = c_index(y, final_model.predict(x))

    joblib.dump(final_model, MODEL_DIR / "rsf_model.joblib")
    x.to_csv(TABLE_DIR / "rsf_design_matrix.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(y).to_csv(TABLE_DIR / "rsf_survival_target.csv", index=False, encoding="utf-8-sig")
    with open(MODEL_DIR / "rsf_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_params": best_params,
                "apparent_c_index": apparent_c,
                "feature_names": list(x.columns),
                "model_output_for_shap": "RSF risk score from RandomSurvivalForest.predict",
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("Best RSF parameters:", best_params)
    print(f"Cross-validated best C-index: {results.iloc[0]['mean_c_index']:.4f}")
    print(f"Apparent C-index on full dataset: {apparent_c:.4f}")
    print(f"Saved RSF model to {MODEL_DIR / 'rsf_model.joblib'}")


if __name__ == "__main__":
    main()
