"""Generate descriptive plots from the processed public datasets."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "outputs" / "figures"
TABLE_DIR = ROOT / "outputs" / "tables"

MODEL_FILE = DATA_DIR / "final_analysis_dataset.xlsx"
TRAFFIC_FILE = DATA_DIR / "traffic_flow_data.xlsx"


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def read_model_data() -> pd.DataFrame:
    return pd.read_excel(MODEL_FILE, sheet_name="Rsf")


def read_speed_data() -> dict[str, pd.Series]:
    sheet_map = {
        "Passenger cars": "乘用车",
        "Heavy vehicles": "重型车",
        "Motorcycles": "摩托车",
    }
    speeds = {}
    for label, sheet in sheet_map.items():
        df = pd.read_excel(TRAFFIC_FILE, sheet_name=sheet)
        speed = pd.to_numeric(df["车速"], errors="coerce").dropna()
        speeds[label] = speed
    return speeds


def plot_speed_distributions() -> None:
    speeds = read_speed_data()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=False)

    summary_rows = []
    for ax, (label, speed) in zip(axes, speeds.items()):
        counts, bins, _ = ax.hist(
            speed,
            bins=20,
            color="#c9c9c9",
            edgecolor="#4d4d4d",
            linewidth=0.7,
        )
        ax.set_title(label)
        ax.set_xlabel("Speed (km/h)")
        ax.set_ylabel("Frequency")
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

        ax2 = ax.twinx()
        cumulative = np.cumsum(counts) / np.sum(counts)
        centers = 0.5 * (bins[:-1] + bins[1:])
        ax2.plot(centers, cumulative, color="#d62728", linewidth=1.6)
        ax2.set_ylim(0, 1.02)
        ax2.set_ylabel("Cumulative frequency")

        summary_rows.append(
            {
                "vehicle_type": label,
                "count": int(speed.count()),
                "mean_speed": float(speed.mean()),
                "p85_speed": float(speed.quantile(0.85)),
                "p95_speed": float(speed.quantile(0.95)),
            }
        )

    fig.tight_layout()
    fig.savefig(FIG_DIR / "traffic_speed_distributions.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(summary_rows).to_csv(
        TABLE_DIR / "traffic_speed_summary.csv", index=False, encoding="utf-8-sig"
    )


def plot_overtaking_duration_distribution() -> None:
    df = read_model_data()
    completed = df[df["overtaking"] == 1].copy()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(
        completed["T"],
        bins=14,
        color="#c9c9c9",
        edgecolor="#4d4d4d",
        linewidth=0.8,
    )
    ax.set_xlabel("Overtaking duration (s)")
    ax.set_ylabel("Frequency")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "overtaking_duration_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def export_vehicle_type_counts() -> None:
    df = read_model_data()
    labels = {0: "Passenger car", 1: "Heavy vehicle", 2: "Motorcycle"}

    overtaken = (
        df["overtaken vehicles"].map(labels).value_counts().rename_axis("vehicle_type").reset_index(name="overtaken_count")
    )
    overtaking = (
        df["overtaking vehicles"].map(labels).value_counts().rename_axis("vehicle_type").reset_index(name="overtaking_count")
    )
    summary = pd.merge(overtaken, overtaking, on="vehicle_type", how="outer").fillna(0)
    summary[["overtaken_count", "overtaking_count"]] = summary[
        ["overtaken_count", "overtaking_count"]
    ].astype(int)
    summary.to_csv(TABLE_DIR / "vehicle_type_counts.csv", index=False, encoding="utf-8-sig")

    x = np.arange(len(summary))
    width = 0.38
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(x - width / 2, summary["overtaken_count"], width, label="Overtaken")
    ax.bar(x + width / 2, summary["overtaking_count"], width, label="Overtaking")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["vehicle_type"])
    ax.set_ylabel("Count")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "vehicle_type_counts.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    plot_speed_distributions()
    plot_overtaking_duration_distribution()
    export_vehicle_type_counts()
    print(f"Saved descriptive figures to {FIG_DIR}")
    print(f"Saved descriptive tables to {TABLE_DIR}")


if __name__ == "__main__":
    main()
