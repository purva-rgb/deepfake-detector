"""Frozen model loaders (loaded once, cached). Xception / Wav2Vec2 are never trained."""
from __future__ import annotations

import time

import numpy as np
import torch

from . import config

_cache: dict = {}


def get_xception():
    """Returns (model, transform_fn, cfg). Fails loudly if XCEPTION_TIMM_NAME does not load."""
    if "xception" in _cache:
        return _cache["xception"]
    import timm

    name = config.XCEPTION_TIMM_NAME
    model = timm.create_model(name, pretrained=True, num_classes=0)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    cfg = dict(model.pretrained_cfg)
    size = int(cfg["input_size"][-1])
    mean = np.array(cfg["mean"], np.float32).reshape(1, 1, 3)
    std = np.array(cfg["std"], np.float32).reshape(1, 1, 3)

    def prep(rgb_uint8: np.ndarray) -> np.ndarray:
        import cv2

        x = cv2.resize(rgb_uint8, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        x = (x - mean) / std
        return x.transpose(2, 0, 1)

    _cache["xception"] = (model, prep, {"size": size, "mean": cfg["mean"], "std": cfg["std"]})
    return _cache["xception"]


@torch.no_grad()
def xception_embed(crops_rgb: list[np.ndarray], batch: int = 16) -> np.ndarray:
    model, prep, _ = get_xception()
    outs = []
    for i in range(0, len(crops_rgb), batch):
        x = torch.from_numpy(np.stack([prep(c) for c in crops_rgb[i : i + batch]]))
        outs.append(model(x).cpu().numpy())
    return np.concatenate(outs, 0).astype(np.float32)


def get_wav2vec2():
    if "w2v" in _cache:
        return _cache["w2v"]
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

    mid = config.WAV2VEC2_MODEL_ID
    fe = Wav2Vec2FeatureExtractor.from_pretrained(mid)
    model = Wav2Vec2Model.from_pretrained(mid)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    _cache["w2v"] = (model, fe)
    return _cache["w2v"]


@torch.no_grad()
def wav2vec2_embed(wave_16k: np.ndarray) -> np.ndarray:
    """Mean-pooled hidden states for layers in config.W2V_LAYERS -> (len(layers), 768)."""
    model, fe = get_wav2vec2()
    inp = fe(wave_16k, sampling_rate=config.SAMPLE_RATE, return_tensors="pt")
    out = model(inp["input_values"], output_hidden_states=True)
    hs = out.hidden_states
    return np.stack([hs[l][0].mean(0).cpu().numpy() for l in config.W2V_LAYERS]).astype(np.float32)


def get_whisper():
    if "whisper" in _cache:
        return _cache["whisper"]
    from faster_whisper import WhisperModel

    _cache["whisper"] = WhisperModel(config.WHISPER_SIZE, device="cpu", compute_type="int8")
    return _cache["whisper"]


def get_face_detector():
    if "face" in _cache:
        return _cache["face"]
    from .faces import FaceDetector

    _cache["face"] = FaceDetector()
    return _cache["face"]


def timed(fn, *a, **k):
    t = time.time()
    r = fn(*a, **k)
    return r, time.time() - t
