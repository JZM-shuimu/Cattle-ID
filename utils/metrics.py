# ============================================================
# 🗂️  utils/metrics.py
# ============================================================
"""Utility helpers for tracking & saving training/validation metrics."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

from config import ARTIFACTS

class MetricTracker:
    """Keeps *lists* of values for each metric name."""

    def __init__(self):
        self.data: Dict[str, List[float]] = defaultdict(list)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self.data[k].append(float(v))

    # --------------------------------------------------------
    # persistence / visualisation
    # --------------------------------------------------------
    def to_csv(self, filename: str | Path):
        df = pd.DataFrame(self.data)
        df.to_csv(ARTIFACTS / filename, index_label="epoch")
        return df

    def plot(self, filename: str | Path, *, dpi: int = 150):
        for k, v in self.data.items():
            plt.figure(dpi=dpi)
            plt.plot(v, label=k)
            plt.xlabel("Epoch")
            plt.ylabel(k)
            plt.title(k)
            plt.legend()
            plt.tight_layout()
            plt.savefig(ARTIFACTS / f"{filename}_{k}.png")
            plt.close()
        return ARTIFACTS / f"{filename}_*.png"
