# ============================================================
# 🗂️  train.py
# ============================================================
"""End‑to‑end training script – run this first."""
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from config import BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY, CHECKPOINTS
from data.datasets import build_dataloaders
from models.cattle_net import CattleNet
from utils.logger import get_logger
from utils.metrics import MetricTracker
import json

from datetime import datetime

LOGGER = get_logger("train")



def main():
    train_loader, val_loader, class_to_idx = build_dataloaders(BATCH_SIZE)

    num_classes = len(class_to_idx)

    model      = CattleNet(num_classes=num_classes).cuda()
    criterion  = nn.CrossEntropyLoss()
    optimizer  = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler  = CosineAnnealingWarmRestarts(optimizer, T_0=10)
    scaler = GradScaler()
    metrics    = MetricTracker()

    best_acc   = 0.0
    for epoch in range(EPOCHS):
        # ---------------------- Training ----------------------
        model.train()
        running_loss = 0.0
        total, correct = 0, 0  # 新增变量用于计算train acc

        for batch_i, (inputs, labels) in enumerate(train_loader, 1):
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                logits = model(inputs, labels)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

            # 计算训练准确率
            preds = logits.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            if batch_i % 50 == 0:
                LOGGER.info("Epoch %3d | Batch %4d/%4d | loss=%.4f | acc=%.4f",
                            epoch + 1, batch_i, len(train_loader),
                            running_loss / batch_i,
                            correct / total)

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total  # 计算最终训练准确率

        # ---------------------- Validation --------------------
        model.eval()
        val_loss   = 0.0
        total, correct = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                with autocast():
                    logits = model(inputs, labels)
                    loss   = criterion(logits, labels)
                val_loss += loss.item()
                preds = logits.argmax(dim=1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc   = correct / total
        LOGGER.info("Epoch %3d COMPLETE | train_loss=%.4f | train_acc=%.4f | val_loss=%.4f | val_acc=%.4f",
                    epoch + 1, train_loss, train_acc, val_loss, val_acc)

        # —— record metrics & save best -----------------------
        metrics.update(train_loss=train_loss, train_acc=train_acc,
                       val_loss=val_loss, val_acc=val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = CHECKPOINTS / "best_model.pth"
            torch.save(model.state_dict(), ckpt_path)
            LOGGER.info("New best model saved to %s (acc=%.4f)", ckpt_path, best_acc)

        scheduler.step()

    # -------------------------- Done -------------------------
    LOGGER.info("Training finished. Best val_acc = %.4f", best_acc)
    # Save metrics & plots ------------------------------------
    df = metrics.to_csv("training_metrics.csv")
    metrics.plot("training_plot")
    LOGGER.info("Metrics saved to %s", df)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    class_idx_file = CHECKPOINTS / f"class_to_idx_{num_classes}.json"
    ckpt_path = CHECKPOINTS / f"best_model.pth"
    
    json.dump(class_to_idx, open(class_idx_file, "w"))

    manifest = {
        "latest_model": ckpt_path.name,
        "latest_class_to_idx": class_idx_file.name,
        "num_classes": num_classes,
        "best_val_acc": best_acc,
        "saved_at": timestamp,
    }
    json.dump(manifest, open(CHECKPOINTS / "manifest.json", "w"), indent=2)


if __name__ == "__main__":
    main()