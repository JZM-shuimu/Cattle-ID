# ============================================================
# 🗂️  models/arcmargin.py
# ============================================================
"""ArcFace (Additive Angular Margin) layer implementation."""
import math
import torch
import torch.nn as nn

class ArcMarginProduct(nn.Module):
    """Implements the *ArcFace* layer from the paper:
    "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
    (https://arxiv.org/abs/1801.07698).

    Args:
        in_features: size of *input* feature vector (embeddings)
        out_features: number of classes
        s: scale factor (default 30.0)
        m: margin (default 0.50)
        easy_margin: use *easy* margin trick (see paper) – disables negative phi.
    """

    def __init__(self, in_features, out_features, *, s=30.0, m=0.50, easy_margin=False):
        super().__init__()
        self.in_features   = in_features
        self.out_features  = out_features
        self.s             = s
        self.m             = m
        self.weight        = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin   = easy_margin
        self.cos_m         = math.cos(m)
        self.sin_m         = math.sin(m)
        self.th            = math.cos(math.pi - m)
        self.mm            = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # normalised features & weights → cosine similarity matrix -------
        cosine = nn.functional.linear(nn.functional.normalize(input), nn.functional.normalize(self.weight))
        sine   = torch.sqrt(1.0 - cosine ** 2 + 1e-6)
        phi    = cosine * self.cos_m - sine * self.sin_m  # cos(θ + m)

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine, device=input.device)
        one_hot.scatter_(1, label.view(-1, 1), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output