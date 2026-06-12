"""Ekstrakcja embeddingów z zamrożonego backbone'u ArcFace (+ opcjonalne TTA)."""
import numpy as np
import torch
import torch.nn.functional as F

from config import BACKBONE_PATH, EMBEDDING_DIM
from models.iresnet import load_backbone

_backbone = None
_device = "cpu"

# normalizacja ArcFace: (px/255 - 0.5) / 0.5  -> [-1, 1]
_MEAN = np.float32([0.5, 0.5, 0.5])
_STD = np.float32([0.5, 0.5, 0.5])


def get_backbone():
    global _backbone
    if _backbone is None:
        _backbone = load_backbone(BACKBONE_PATH, num_features=EMBEDDING_DIM, device=_device)
    return _backbone


def _to_tensor(rgb_uint8):
    """RGB HxWx3 uint8 -> tensor 1x3x112x112 znormalizowany."""
    x = rgb_uint8.astype(np.float32) / 255.0
    x = (x - _MEAN) / _STD
    x = np.transpose(x, (2, 0, 1))[None]
    return torch.from_numpy(x).to(_device)


@torch.no_grad()
def embed_aligned(rgb_uint8, tta=False):
    """Aligned crop 112x112 RGB -> embedding 512D (L2-znormalizowany).

    tta=True: uśrednia embedding oryginału i odbicia poziomego (test-time aug).
    """
    model = get_backbone()
    emb = model(_to_tensor(rgb_uint8))
    if tta:
        flipped = rgb_uint8[:, ::-1, :].copy()
        emb = emb + model(_to_tensor(flipped))
    return F.normalize(emb, dim=1).squeeze(0).cpu().numpy()


def cosine_distance(e1, e2):
    return float(1.0 - np.dot(e1, e2))


def cosine_similarity(e1, e2):
    return float(np.dot(e1, e2))
