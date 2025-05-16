"""Backbone ConvNeXt‑Large + ArcMargin head."""
import torch.nn as nn
import torchvision.models as models
# import torch.nn.functional as F
import torch
from .arcmargin import ArcMarginProduct
from config import ARC_S, ARC_M

import timm
from iresnet import iresnet50  # 确保 iresnet.py 在项目目录中


class CattleNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        # self.features = iresnet50(pretrained=True)  # 初始化模型结构
        # # self.features.load_state_dict(torch.load('model_ir_se50.pth'))  # 加载预训练权重
        # self.embedding = nn.Linear(512, 512)

        backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        self.features = nn.Sequential(backbone.features, backbone.avgpool)  # 1536‑D output
        self.embedding = nn.Linear(768, 512)

        # backbone = models.convnext_large(weights=models.ConvNeXt_Large_Weights.DEFAULT)
        # self.features = nn.Sequential(backbone.features, backbone.avgpool)  # 1536Large output 768Tiny
        # self.embedding = nn.Linear(1536, 512)

        # backbone = models.resnet18(pretrained=True)
        # self.features = nn.Sequential(*list(backbone.children())[:-1])
        # self.embedding = nn.Linear(512, 512)

        # backbone = models.resnet50(pretrained=True)
        # self.features = nn.Sequential(*list(backbone.children())[:-1])
        # self.embedding = nn.Linear(2048, 512)

        # backbone = timm.create_model('vit_tiny_patch16_224', pretrained=True)
        # self.patch_embed = backbone.patch_embed
        # self.cls_token = backbone.cls_token
        # self.pos_embed = backbone.pos_embed
        # self.pos_drop = backbone.pos_drop
        # self.blocks = backbone.blocks
        # self.norm = backbone.norm
        # self.embedding = nn.Linear(192, 512)

        # backbone = timm.create_model('vit_small_patch16_224', pretrained=True)
        # self.patch_embed = backbone.patch_embed
        # self.cls_token = backbone.cls_token
        # self.pos_embed = backbone.pos_embed
        # self.pos_drop = backbone.pos_drop
        # self.blocks = backbone.blocks
        # self.norm = backbone.norm
        # self.embedding = nn.Linear(384, 512)

        # backbone = models.mobilenet_v2(pretrained=True)
        # self.features = backbone.features
        # self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.embedding = nn.Linear(1280, 512)

        self.arc_head  = ArcMarginProduct(512, num_classes, s=ARC_S, m=ARC_M)

    def forward(self, x, labels=None):
        # all the rest
        feat = self.features(x)
        feat = feat.flatten(1)

        # mobilenet-v2
        # feat = self.features(x)
        # feat = self.pool(feat).view(feat.size(0), -1)            # mobilenet-v2

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
        return self.arc_head(feat, labels) if labels is not None else feat


    # 添加方法
    def extract_features(self, x):

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

