"""M0: load every model once, print load time and output shape."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src import config, models


def main():
    print("DATASET_ROOT:", config.DATASET_ROOT, "exists:", config.DATASET_ROOT.exists())
    print("XCEPTION_TIMM_NAME:", config.XCEPTION_TIMM_NAME)
    print("WAV2VEC2_MODEL_ID:", config.WAV2VEC2_MODEL_ID)
    print("WHISPER_SIZE:", config.WHISPER_SIZE)
    ok = True

    # Xception (STOP on failure, no substitution)
    t = time.time()
    try:
        model, prep, cfg = models.get_xception()
        dt = time.time() - t
        x = np.random.randint(0, 255, (cfg["size"], cfg["size"], 3), np.uint8)
        t2 = time.time()
        e = models.xception_embed([x, x])
        print(f"[xception] loaded {config.XCEPTION_TIMM_NAME} in {dt:.1f}s; cfg={cfg}; "
              f"embedding shape={e.shape}; infer(2 crops)={time.time()-t2:.2f}s")
    except Exception as ex:  # noqa: BLE001
        print(f"[xception] FAILED to load {config.XCEPTION_TIMM_NAME!r}: {type(ex).__name__}: {ex}")
        print("STOP: not substituting another model.")
        sys.exit(2)

    t = time.time()
    try:
        models.get_wav2vec2()
        dt = time.time() - t
        t2 = time.time()
        e = models.wav2vec2_embed(np.random.randn(16000 * 4).astype(np.float32) * 0.1)
        print(f"[wav2vec2] loaded {config.WAV2VEC2_MODEL_ID} in {dt:.1f}s; layers={config.W2V_LAYERS}; "
              f"embedding shape={e.shape}; infer(4s)={time.time()-t2:.2f}s")
    except Exception as ex:  # noqa: BLE001
        ok = False
        print(f"[wav2vec2] FAILED: {type(ex).__name__}: {ex}")

    t = time.time()
    try:
        w = models.get_whisper()
        dt = time.time() - t
        t2 = time.time()
        segs, info = w.transcribe(np.zeros(16000 * 2, np.float32), language="en")
        segs = list(segs)
        print(f"[whisper] loaded faster-whisper {config.WHISPER_SIZE} int8 cpu in {dt:.1f}s; "
              f"transcribe(2s silence) -> {len(segs)} segments in {time.time()-t2:.2f}s")
    except Exception as ex:  # noqa: BLE001
        ok = False
        print(f"[whisper] FAILED: {type(ex).__name__}: {ex}")

    t = time.time()
    try:
        fd = models.get_face_detector()
        dt = time.time() - t
        img = np.zeros((240, 320, 3), np.uint8)
        r = fd.detect_largest(img)
        print(f"[face] backend={fd.backend} loaded in {dt:.1f}s; blank image -> {r}; note={fd.note!r}")
    except Exception as ex:  # noqa: BLE001
        ok = False
        print(f"[face] FAILED: {type(ex).__name__}: {ex}")

    print("SMOKE OK" if ok else "SMOKE: some components failed (see above)")


if __name__ == "__main__":
    main()
