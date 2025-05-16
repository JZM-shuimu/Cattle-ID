# ============================================================
# (optional) 🗂️  plots.py – quick viewer for saved CSV metrics
# ============================================================
"""Utility to open *training_metrics.csv* and re‑plot curves."""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from config import ARTIFACTS


def main(csv: str | Path = ARTIFACTS / "training_metrics.csv"):
    df = pd.read_csv(csv, index_col=0)
    for col in df.columns:
        plt.figure()
        plt.plot(df[col])
        plt.title(col)
        plt.xlabel("Epoch")
        plt.ylabel(col)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
