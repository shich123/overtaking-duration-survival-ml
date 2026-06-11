"""Prepare the final overtaking-duration analysis dataset.

This script reads the processed Excel workbook used for the manuscript analyses
and exports reproducible CSV files for survival modeling.
"""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs" / "tables"

SOURCE_FILE = DATA_DIR / "final_analysis_dataset.xlsx"
MODEL_SHEET = "Rsf"

CONTINUOUS_FEATURES = ["D", "ID", "V1", "V2", "DV", "H", "VL"]
CATEGORICAL_FEATURES = ["INVASION", "overtaken vehicles", "overtaking vehicles"]
TIME_COL = "T"
EVENT_COL = "overtaking"


def load_model_data() -> pd.DataFrame:
    """Load and validate the final RSF/AFT analysis dataset."""
    df = pd.read_excel(SOURCE_FILE, sheet_name=MODEL_SHEET)
    expected = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES + [EVENT_COL, TIME_COL]
    missing = [col for col in expected if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in {SOURCE_FILE.name}: {missing}")

    df = df.loc[:, expected].copy()
    df[EVENT_COL] = df[EVENT_COL].astype(int)
    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="raise")
    for col in CONTINUOUS_FEATURES + CATEGORICAL_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="raise")

    if df.isna().any().any():
        raise ValueError("The modeling dataset contains missing values.")
    if not set(df[EVENT_COL].unique()).issubset({0, 1}):
        raise ValueError(f"{EVENT_COL} must be coded as 0/1.")

    return df


def summarize_modeling_data(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize the final modeling sample."""
    n_events = int(df[EVENT_COL].sum())
    n_total = len(df)
    return pd.DataFrame(
        [
            {
                "dataset": "final survival-analysis dataset",
                "total_observations": n_total,
                "complete_maneuvers": n_events,
                "right_censored_observations": n_total - n_events,
            }
        ]
    )


def write_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """Write CSV using a plain text write as a fallback-friendly method."""
    path.write_text(df.to_csv(index=index), encoding="utf-8-sig")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_model_data()

    model_csv = OUT_DIR / "prepared_modeling_dataset.csv"
    summary_csv = OUT_DIR / "dataset_summary.csv"
    variable_csv = OUT_DIR / "variable_summary.csv"

    write_csv(df, model_csv, index=False)
    write_csv(summarize_modeling_data(df), summary_csv, index=False)
    write_csv(df.describe(include="all").T, variable_csv, index=True)

    print(f"Saved modeling dataset: {model_csv}")
    print(f"Saved dataset summary: {summary_csv}")
    print(f"Saved variable summary: {variable_csv}")
    print(f"Rows: {len(df)}; events: {int(df[EVENT_COL].sum())}; censored: {int((1 - df[EVENT_COL]).sum())}")


if __name__ == "__main__":
    main()
