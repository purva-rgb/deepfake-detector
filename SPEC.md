# SPEC.md: Deepfake Scam Ad Detector (PS4 "Is That Really Them?")

Trimmed 5-hour build scope. This file is the source of truth. The long
architecture document is reference only. If the two disagree, this file
wins.

## 1. Goal

Analyse a 15-90 second video ad (English; Hindi demo only) and output:

-   Overall risk: LOW, MEDIUM, HIGH or INSUFFICIENT EVIDENCE
-   Deepfake evidence score D, scam evidence score S, optional AV
    mismatch M
-   Reliability per modality (R_v, R_a, R_t) with warnings
-   Top 5 timestamped evidence cards (thumbnail, transcript or OCR
    snippet, plain-language reason)
-   Scam checklist (6 categories)

Scores are heuristic evidence scores, not probabilities. The UI must say
so.

Deepfake risk and scam risk are computed independently and combined only
at the end. A real video can be a scam, and a fake video may not be a
scam.

## 2. Frozen decisions

  -----------------------------------------------------------------------
  Role                                Choice
  ----------------------------------- -----------------------------------
  Visual features                     Xception (frozen), pooled embedding
                                      of face crops

  Audio features                      Wav2Vec2 (frozen), mean-pooled
                                      hidden states from a middle layer

  Heads                               Logistic regression per modality,
                                      trained on cached embeddings

  Sync (optional)                     MAR + audio RMS envelope +
                                      cross-correlation (+/-200 ms). No
                                      SyncNet.

  ASR                                 faster-Whisper base, int8, CPU

  OCR                                 EasyOCR (en, hi), 1 frame every 2 s

  Scam                                Deterministic YAML rule engine with
                                      context guard. No neural
                                      classifier.

  Faces                               MediaPipe Face Landmarker

  UI                                  Streamlit

  Input                               Upload (mp4/mov/webm) or public URL
                                      via yt-dlp
  -----------------------------------------------------------------------

Explicitly OUT of scope: model fine-tuning, DCT/boundary auxiliary
features, ablation study, WhatsApp bot, SyncNet, LLM-based decisions.

## 3. Data

-   **Training/calibration:** FakeAVCeleb subset, about 1,600 clips
    (about 400 each of RVRA, RVFA, FVRA, FVFA), English only. Reduce to
    about 1,000 if the benchmark shows extraction is too slow.
-   **Split:** grouped by speaker/source identity (`GroupKFold` or
    `LeaveOneGroupOut`). A speaker must never appear on both sides of a
    split.
-   **Labels:** visual label = 1 if the video is fake (FV*), audio label
    = 1 if the audio is fake (*FA).
-   **Hindi clips (17):** held-out demo set. NEVER used for training,
    calibration, thresholds or rule tuning. Report per-clip D, S, M,
    R_v, R_a, R_t. No accuracy claim.
-   **Scam text:** the Hinglish/English scam dataset plus 20-30
    hand-written hard negatives, for evaluating rules only.

## 4. Pipeline

1.  **Ingest:** validate format, download if URL, enforce a 90 s cap,
    hash the file (cache key), FFmpeg to frames and 16 kHz mono wav.
2.  **Segment:** 4 s windows, 2 s hop.
3.  **Visual (per window):** MediaPipe landmarks at about 10-12 fps. Up
    to 4 face crops per window at 1-2 fps, with 25% margin, resized to
    224x224. Xception embedding averaged over crops, then logistic head
    gives p_fake per crop. `S_visual = 0.5*mean(p) + 0.5*max(p)`.
4.  **Audio (per window):** Wav2Vec2 embedding, then logistic head gives
    `S_audio`.
5.  **Sync (optional, per window):** MAR(t) and RMS A(t), z-score,
    cross-correlation over +/-200 ms. Normalise against RVRA reference
    distribution to get M in \[0,1\]. Only run if the preconditions in
    section 6 hold. Otherwise M = N/A.
6.  **Text:** faster-Whisper transcript with timestamps, and EasyOCR
    text. Normalise lightly (case, punctuation, whitespace). Do NOT
    strip meaningful Hindi/Hinglish words.
7.  **Scam rules:** run on the combined text per window, giving S_scam
    and R_t.
8.  **Fusion, aggregation, risk band, evidence ranking** (sections 5-8).
9.  **UI.**

## 5. Reliability

All reliabilities are in \[0,1\].

-   **R_v:** from face detection confidence, face size relative to
    frame, blur (variance of Laplacian), visible fraction, pose. Tiny,
    blurred or extreme-pose faces give a low R_v. No face gives R_v = 0.
-   **R_a:** from speech ratio (VAD), clipping, silence, optional music
    dominance.
-   **R_t:** from Whisper average log-prob or confidence and OCR
    confidence.
-   **R_sync:** valid only when the preconditions below hold.

**Gate:** if R \< 0.15 for a modality, drop it (mark N/A). Never
substitute 0 and never pretend it was evaluated.

**Sync preconditions:** at least 70% usable face frames, adequate face
size, speech ratio \>= 0.4, R_a \>= 0.3, one dominant face. Otherwise M
= N/A. Dubbed or voice-over content can legitimately mismatch, so show a
context warning and never present M as proof.

## 6. Fusion (per window)

``` text
D = (R_v*S_visual + R_a*S_audio) / (R_v + R_a)        # drop any dropped modality's term and weight
R_media = 0.6*D + 0.4*M   if M available else D
R = max(S_scam, D, R_media)
if D > 0.65 and S_scam > 0.65: R = min(1.0, R + 0.10)   # synergy
```

**Low-reliability cap (new):** let `R_sum = R_v + R_a` over surviving
modalities. If `R_sum < 0.5`, cap the deepfake contribution (D and
R_media) at 0.64, so weak evidence alone can never produce HIGH. A
strong scam score S can still produce HIGH.

**Insufficient evidence:** if `R_v + R_a < 0.30` (media unusable) and
`S_video < 0.35`, the overall result is INSUFFICIENT EVIDENCE, never
LOW.

### 6a. Scam score

-   `S_max` = maximum matched rule strength in the window.
-   `S_scam = min(1, S_max + 0.10*(N_categories - 1))`.
-   Categories: (1) financial promise, (2) urgency, (3) authority
    impersonation, (4) payment/credentials, (5) off-platform
    redirect, (6) fake branding.
-   **Context guard:** a keyword alone is never a signal. "Never share
    your OTP with anyone" must NOT match category 4. Use negation and
    warning patterns (never, do not, beware, don't share, avoid) within
    a short token distance to suppress a match. Keep the guard in
    `rules.yaml` as data plus a small, tested matcher.
-   Each rule has: id, category, regex or phrase list (English, Hindi in
    Devanagari, Hinglish in Roman script), strength in \[0,1\], optional
    `suppress_if` patterns.

## 7. Video-level aggregation

-   `D_video` = mean of top 3 windows by D (top 25% of windows if more
    than 12 windows).
-   `S_video` = max window S_scam.
-   `M_video` = mean of top 3 valid sync windows, N/A if fewer than 2
    valid windows.
-   `R_video = max(S_video, D_video, R_media_video)` with synergy and
    the cap rules above applied at video level.

**Bands:** LOW if R \< 0.35, MEDIUM if 0.35 \<= R \< 0.65, HIGH if R \>=
0.65.

## 8. Evidence engine

Each detector emits records:
`{t_start, t_end, modality, detector, score, reliability, text, thumbnail_path, reason}`.

-   Rank by `score * reliability`, merge adjacent intervals, dedupe,
    keep the top 5.
-   Reasons use plain language (for example "Mouth movement doesn't
    match speech at 0:07-0:12").

## 9. UI (Streamlit)

Upload or URL box, step-by-step progress, risk banner, three score bars
(Deepfake, Scam, AV mismatch, with N/A shown explicitly), reliability
panel with warnings, timeline of windows coloured by risk, evidence
cards, scam checklist, transcript with highlighted suspicious spans,
footer: "Scores are heuristic evidence scores, not probabilities.
Results depend on video quality, face visibility, audio quality,
compression, language and model coverage."

## 10. Failure handling

Every model call is wrapped. On failure: set that modality to N/A, add a
warning, continue. The pipeline must never crash because one modality
failed.

## 11. Files

``` text
project/
  app.py  pipeline.py  config.py  models.py
  ingest.py  segment.py  visual.py  audio.py  sync.py
  semantic.py  reliability.py  fusion.py  evidence.py  calibrate.py
  rules.yaml
  scripts/  smoke_test.py  extract_features.py  train_heads.py
  tests/    test_fusion.py  test_scam_rules.py  test_reliability.py
  models/  cache/  data/
```

`config.py` holds every threshold above plus model IDs. Model IDs are
filled in by the human after verification (see `.cursorrules`).

## 12. Build order (one step per prompt, test before moving on)

0.  `scripts/smoke_test.py`: load Xception, Wav2Vec2, Whisper, EasyOCR,
    MediaPipe. Report load time, per-item inference time, output shape,
    memory. Run on 10 clips and print the projected total time for N
    clips.
1.  `ingest.py` (+ yt-dlp URL support)
2.  `segment.py`
3.  `scripts/extract_features.py`: parallel (2-3 workers), resumable,
    writes one `.npz` per clip with label, source_id, dataset tag. Skips
    clips already done.
4.  `scripts/train_heads.py`: StandardScaler + LogisticRegression
    (`class_weight="balanced"`, small C), grouped CV, report out-of-fold
    AUC with bootstrap CI, save with joblib.
5.  `semantic.py` + `rules.yaml` + tests for scam rules, including hard
    negatives.
6.  `reliability.py`, `fusion.py`, `evidence.py` + unit tests with
    hand-computed numbers.
7.  `sync.py` (optional; keep only if RVRA vs fake shows separation,
    otherwise disable and say so).
8.  `pipeline.py` and `app.py`.
9.  Run the 17 Hindi clips plus the self-recorded 30 s clip. Log
    per-module latency.

## 13. Acceptance checks

-   Fusion tests pass: missing modality removes its term, the cap rule
    works, the synergy bonus applies once, INSUFFICIENT EVIDENCE
    triggers when expected.
-   Scam tests pass: "Never share your OTP" does not match, "send your
    OTP now" does match, and legitimate WhatsApp support text does not
    trigger redirect.
-   A 30-60 s clip finishes in about 1-2 minutes on CPU, with latency
    logged per module.
-   The UI shows N/A, not 0, for any unavailable modality.
-   Reported AUCs come from grouped, out-of-fold predictions, with the
    caveat that they are indicative.
