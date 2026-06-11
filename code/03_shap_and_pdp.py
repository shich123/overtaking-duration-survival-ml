"""Generate SHAP and partial-dependence outputs for the fitted RSF model.

SHAP values are computed for the RSF risk score produced by
RandomSurvivalForest.predict. Positive SHAP values therefore indicate an
increase in the predicted RSF risk score, not directly an increase in duration.
If the manuscript interprets positive SHAP values as longer duration, first
replace the model output function below with a duration-derived prediction.
"""

from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
FIG_DIR = ROOT / "outputs" / "figures"
MODEL_DIR = ROOT / "outputs" / "models"


def load_inputs():
    model = joblib.load(MODEL_DIR / "rsf_model.joblib")
    x = pd.read_csv(TABLE_DIR / "rsf_design_matrix.csv")
    with open(MODEL_DIR / "rsf_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return model, x, metadata


def manual_pdp(model, x: pd.DataFrame, feature: str, grid_size: int = 30) -> pd.DataFrame:
    """Compute a one-way PDP for the RSF risk score."""
    values = x[feature]
    if values.nunique() <= 10:
        grid = np.sort(values.unique())
    else:
        grid = np.linspace(values.quantile(0.05), values.quantile(0.95), grid_size)

    rows = []
    x_tmp = x.copy()
    for value in grid:
        x_tmp[feature] = value
        rows.append({"feature": feature, "value": value, "mean_risk_score": model.predict(x_tmp).mean()})
    return pd.DataFrame(rows)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    model, x, metadata = load_inputs()

    def predict_risk(x_array):
        x_df = pd.DataFrame(x_array, columns=x.columns)
        return model.predict(x_df)

    masker = shap.maskers.Independent(x, max_samples=min(50, len(x)))
    explainer = shap.Explainer(predict_risk, masker, algorithm="permutation")
    shap_values = explainer(x, max_evals=2 * x.shape[1] + 1)

    shap_df = pd.DataFrame(shap_values.values, columns=x.columns)
    shap_df.to_csv(TABLE_DIR / "rsf_shap_values.csv", index=False, encoding="utf-8-sig")

    importance = (
        shap_df.abs()
        .mean(axis=0)
        .sort_values(ascending=False)
        .rename("mean_abs_shap")
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    importance.to_csv(TABLE_DIR / "rsf_shap_importance.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(8, 6))
    shap.plots.beeswarm(shap_values, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "rsf_shap_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close()

    pdp_frames = []
    for feature in x.columns:
        pdp_frames.append(manual_pdp(model, x, feature))
    pdp = pd.concat(pdp_frames, ignore_index=True)
    pdp.to_csv(TABLE_DIR / "rsf_partial_dependence.csv", index=False, encoding="utf-8-sig")

    top_features = importance["feature"].head(9).tolist()
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    for ax, feature in zip(axes.ravel(), top_features):
        sub = pdp[pdp["feature"] == feature]
        ax.plot(sub["value"], sub["mean_risk_score"], marker="o", linewidth=1.5)
        ax.set_title(feature)
        ax.set_xlabel(feature)
        ax.set_ylabel("Mean RSF risk score")
    for ax in axes.ravel()[len(top_features):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "rsf_partial_dependence_top_features.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Generated SHAP values, SHAP importance, beeswarm plot, and PDP outputs.")
    print("Important: SHAP values explain the RSF risk score unless the prediction function is changed.")


if __name__ == "__main__":
    main()
