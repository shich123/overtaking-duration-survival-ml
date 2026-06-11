"""Fit the log-logistic accelerated failure time benchmark model.

The AFT model is used as the transparent parametric benchmark for the RSF
model. The variable set follows the benchmark specification reported in the
manuscript.
"""

from pathlib import Path

import pandas as pd
from lifelines import LogLogisticAFTFitter
from lifelines.utils import concordance_index


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
MODEL_DIR = ROOT / "outputs" / "models"

TIME_COL = "T"
EVENT_COL = "overtaking"
BENCHMARK_COLUMNS = [
    TIME_COL,
    EVENT_COL,
    "ID",
    "VL",
    "INVASION",
    "overtaken vehicles",
    "overtaking vehicles",
    "V1",
    "V2",
]
MANUSCRIPT_NAMES = {
    "VL": "L",
    "INVASION": "Oncoming traffic",
    "overtaken vehicles": "Overtaken vehicles",
    "overtaking vehicles": "Overtaking vehicles",
    "V1": "v1",
    "V2": "v2",
}


def load_data() -> pd.DataFrame:
    path = TABLE_DIR / "prepared_modeling_dataset.csv"
    if not path.exists():
        raise FileNotFoundError("Run code/00_prepare_data.py before fitting models.")
    return pd.read_csv(path)


def select_benchmark_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Select and rename variables used in the manuscript AFT benchmark."""
    missing = [col for col in BENCHMARK_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for AFT benchmark model: {missing}")
    return df.loc[:, BENCHMARK_COLUMNS].rename(columns=MANUSCRIPT_NAMES)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = select_benchmark_variables(load_data())
    aft = LogLogisticAFTFitter()
    aft.fit(df, duration_col=TIME_COL, event_col=EVENT_COL)

    summary = aft.summary.copy()
    summary.to_csv(TABLE_DIR / "aft_log_logistic_summary.csv", encoding="utf-8-sig")

    # Lifelines AFT predictions are median duration outputs. Higher predicted
    # duration should correspond to longer observed duration.
    pred_median = aft.predict_median(df)
    c_index = concordance_index(df[TIME_COL], pred_median, df[EVENT_COL])
    pd.DataFrame(
        [{"model": "Log-logistic AFT", "c_index_apparent": c_index}]
    ).to_csv(TABLE_DIR / "aft_c_index.csv", index=False, encoding="utf-8-sig")

    aft.print_summary()
    print(f"Apparent C-index: {c_index:.4f}")
    print(f"Saved AFT summary to {TABLE_DIR / 'aft_log_logistic_summary.csv'}")


if __name__ == "__main__":
    main()
