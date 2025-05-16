# ============================================================
# 🗂️  evaluate.py
# ============================================================
"""Per‑class evaluation after training."""
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from config import CHECKPOINTS, ARTIFACTS
from data.datasets import BalancedDataset, CattleDataset
from models.cattle_net import CattleNet
from utils.logger import get_logger

LOGGER = get_logger("evaluate")

def main(model_path: str | Path | None = None):
    model_path = Path(model_path or CHECKPOINTS / "best_model.pth")
    ds         = BalancedDataset()
    val_ds     = CattleDataset(ds.val_files, train=False)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=8)

    idx_to_class = {v: k for k, v in val_ds.class_to_idx.items()}
    model = CattleNet(num_classes=len(idx_to_class))
    model.load_state_dict(torch.load(model_path, map_location="cuda"))
    model = model.cuda().eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.cuda()
            logits = model(inputs, labels.cuda())
            preds  = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())

    # ---------------- Aggregate metrics ----------------------
    report = classification_report(all_labels, all_preds, target_names=[idx_to_class[i] for i in range(len(idx_to_class))], output_dict=True)
    cm     = confusion_matrix(all_labels, all_preds)

    # Save to CSV / log text ----------------------------------
    report_df = pd.DataFrame(report).transpose()
    cm_df     = pd.DataFrame(cm, index=[idx_to_class[i] for i in range(len(idx_to_class))], columns=[idx_to_class[i] for i in range(len(idx_to_class))])

    report_csv = ARTIFACTS / "classification_report.csv"
    cm_csv     = ARTIFACTS / "confusion_matrix.csv"
    report_df.to_csv(report_csv)
    cm_df.to_csv(cm_csv)

    LOGGER.info("Classification report saved to %s", report_csv)
    LOGGER.info("Confusion matrix   saved to %s", cm_csv)

if __name__ == "__main__":
    main()