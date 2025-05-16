import os
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import pandas as pd
import shutil
import json

from config import CHECKPOINTS, IMG_SIZE, ARTIFACTS
from models.cattle_net import CattleNet
from utils.logger import get_logger

logger = get_logger("retrieval")

# ===================== 参数 =====================
QUERY_DIR   = Path("query")    # ← 替换为真实路径
GALLERY_DIR = Path("gallery")  # ← 替换为真实路径
BATCH_SIZE  = 8
TOP_K       = 5

MANIFEST_PATH = CHECKPOINTS / "manifest.json"
with open(MANIFEST_PATH) as f:
    manifest = json.load(f)
    NUM_CLASSES = manifest['num_classes']

MODEL_PATH     = CHECKPOINTS / manifest["latest_model"]
CLASS_IDX_PATH = CHECKPOINTS / manifest["latest_class_to_idx"]

# ===================== 预处理 =====================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def load_images_from_folder(folder):
    paths = sorted([p for p in Path(folder).glob("*") if p.suffix.lower() in [".jpg", ".png", ".jpeg"]])
    data = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        tensor = transform(img)
        data.append((str(p), tensor))
    return data

# 全局统计变量
total_feat_time = 0
total_feat_images = 0

@torch.no_grad()
def extract_features_with_stats(model, image_data, label=""):
    global total_feat_time, total_feat_images
    start = time.time()

    paths, tensors = zip(*image_data)
    tensors = torch.stack(tensors).cuda()
    feats = []

    for i in range(0, len(tensors), BATCH_SIZE):
        batch = tensors[i:i+BATCH_SIZE]
        emb = model.extract_features(batch)
        emb = F.normalize(emb, dim=1)
        feats.append(emb.cpu())

    end = time.time()
    elapsed = end - start
    fps = len(image_data) / elapsed

    total_feat_time += elapsed
    total_feat_images += len(image_data)

    print(f"\n️ [{label} 特征提取] 图像数量: {len(image_data)}")
    print(f"️ [{label} 特征提取] 耗时: {elapsed:.2f} 秒")
    print(f" [{label} 特征提取] FPS: {fps:.2f} frames/sec\n")

    return list(paths), torch.cat(feats, dim=0)

def compute_topk(query_feats, gallery_feats, gallery_paths, topk=5):
    sims = torch.mm(query_feats, gallery_feats.T)
    topk_vals, topk_indices = sims.topk(topk, dim=1)
    return topk_vals, topk_indices

def build_html_output(results, html_path="retrieval_results.html"):
    html = ["<html><body><h1>牛脸检索结果</h1><table border='1'>"]
    for entry in results:
        html.append("<tr>")
        html.append(f"<td><b>Query:</b><br><img src='{entry['query']}' height='160'></td>")
        for match in entry["matches"]:
            html.append(f"<td><img src='{match['path']}' height='160'><br>Score: {match['score']:.4f}</td>")
        html.append("</tr>")
    html.append("</table></body></html>")
    with open(ARTIFACTS / html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    logger.info("HTML 结果已保存至 %s", ARTIFACTS / html_path)

def run_retrieval(repeat_times=10):
    logger.info("加载模型")
    model = CattleNet(num_classes=NUM_CLASSES).cuda()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    def extract_features_method(self, x):
        # all the rest
        feat = self.features(x)
        feat = feat.flatten(1)

        # mobilenet-v2
        # feat = self.features(x)
        # feat = F.adaptive_avg_pool2d(feat, (1, 1))  # [B, 1280, 1, 1]
        # feat = feat.view(feat.size(0), -1)  # [B, 1280]

        # vit
        # B = x.size(0)
        # x = self.patch_embed(x)  # 等价于 self.features[0](x)
        # cls_token = self.cls_token.expand(B, -1, -1)
        # x = torch.cat((cls_token, x), dim=1)
        # x = x + self.pos_embed[:, :x.size(1), :]
        # x = self.pos_drop(x)
        # x = self.blocks(x)
        # x = self.norm(x)
        # feat = x[:, 0]  # 取 cls token

        feat = self.embedding(feat)
        return feat
    model.extract_features = extract_features_method.__get__(model)

    logger.info("加载 query 图像")
    query_data = load_images_from_folder(QUERY_DIR)
    logger.info("加载 gallery 图像")
    gallery_data = load_images_from_folder(GALLERY_DIR)

    if not query_data or not gallery_data:
        logger.error("查询或库图像为空")
        return

    global total_feat_time, total_feat_images
    total_feat_time = 0
    total_feat_images = 0

    for run in range(repeat_times):
        print(f"\n=====  第 {run+1}/{repeat_times} 次执行 =====")
        query_paths, query_feats = extract_features_with_stats(model, query_data, label=f"Query#{run+1}")
        gallery_paths, gallery_feats = extract_features_with_stats(model, gallery_data, label=f"Gallery#{run+1}")

    # 最后一次执行匹配并输出
    logger.info("开始匹配 (最后一次)，每个 query 输出 top-%d 结果", TOP_K)
    topk_vals, topk_indices = compute_topk(query_feats, gallery_feats, gallery_paths, topk=TOP_K)

    results = []
    csv_records = []
    for i, q_path in enumerate(query_paths):
        match_entries = []
        for rank, (idx, score) in enumerate(zip(topk_indices[i], topk_vals[i])):
            g_path = gallery_paths[idx]
            match_entries.append({"path": g_path, "score": score.item()})
            csv_records.append({
                "query": q_path,
                "rank": rank + 1,
                "match": g_path,
                "score": score.item()
            })
        results.append({"query": q_path, "matches": match_entries})

    df = pd.DataFrame(csv_records)
    csv_path = ARTIFACTS / "retrieval_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info(" CSV 结果已保存至 %s", csv_path)

    build_html_output(results, "retrieval_results.html")

    if total_feat_images > 0:
        avg_time_per_image = total_feat_time / total_feat_images
        fps = total_feat_images / total_feat_time
        print(f"\n [10 次特征提取平均统计]")
        print(f"️ 图像总数: {total_feat_images}")
        print(f"️ 总耗时: {total_feat_time:.2f} 秒")
        print(f" 每张图平均耗时: {avg_time_per_image:.4f} 秒")
        print(f" 平均 FPS: {fps:.2f} frames/sec")

if __name__ == "__main__":
    run_retrieval(repeat_times=1)
