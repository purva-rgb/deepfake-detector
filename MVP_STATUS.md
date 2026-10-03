# MVP_STATUS

Prototype of SPEC.md (frozen Xception + Wav2Vec2, LR heads, reliability-aware fusion, rule-based scam engine, Streamlit).
Cuts as agreed: English only, no OCR, no AV sync (M = N/A), no URL input, no test split, no bootstrap CIs.
Environment: Windows, Python 3.14.7, CPU only. Everything below was produced by real runs in this session (2026-10-03).

## Components

| Component | Status | Run command / notes |
|---|---|---|
| Model loading (M0) | WORKING | `python scripts/smoke_test.py` - legacy_xception (2048-d, 299x299, mean/std 0.5) 1.4 s warm; wav2vec2-base 3 layers x 768 (infer 0.7 s / 4 s); faster-whisper base int8 (infer 0.5 s); MediaPipe face detector |
| Data manifest (M1) | WORKING | `python scripts/prepare_data.py` -> `data/subset.csv` (160 clips, manifests only, nothing copied) |
| Feature extraction (M2) | WORKING | `python scripts/extract.py --workers 2` (resumable). 160/160 clips, 0 failures, 249 s wall (1.6 s/clip with 2 workers) |
| Face detection | WORKING (MediaPipe) | MediaPipe Tasks `FaceDetector` (BlazeFace short-range). **No Haar fallback was needed.** SPEC names the Face Landmarker; landmarks are not used, so the detector was used instead |
| Audio quality (VAD etc.) | WORKING | webrtcvad (`webrtcvad-wheels`) installed fine; energy-VAD fallback exists but was not used. SNR-like score = 10*log10(P90/P10) of 20 ms frame energies (proxy, not true SNR) |
| Visual head (Xception + LR) | PARTIAL | Works; weak (val AUC 0.72). See metrics |
| Audio head (Wav2Vec2 + LR) | WORKING, caveat | val AUC 1.00 on 32 clips - do NOT read as generalisation (see caveats) |
| Scam rules (`rules.yaml`, `src/scam.py`) | WORKING | 6 English categories, >= 6 patterns each, context guard (negation within 8 tokens, same sentence) + `suppress_if` |
| Transcript (`src/semantic.py`) | WORKING | faster-whisper base, int8, CPU, `language="en"` |
| Reliability (`src/reliability.py`) | WORKING | Simple documented formulas (uncalibrated heuristics) |
| Fusion (`src/fusion.py`) | WORKING | SPEC 6-7: reliability-weighted D, cap, synergy once, INSUFFICIENT rule, top-3/25% video aggregation |
| Pipeline (`src/pipeline.py`) | WORKING | `python scripts/run_pipeline_cli.py c032 c152` (clip ids from subset.csv, or a path) |
| Streamlit UI (`app.py`) | WORKING (render-tested, upload not browser-tested) | `streamlit run app.py` -> http://localhost:8501. Server start confirmed (health `ok`, HTTP 200). Report rendering tested with Streamlit `AppTest`. A real browser file-upload click-through was NOT performed |
| AV mismatch (M) | UNAVAILABLE (cut) | Always N/A, shown explicitly in UI |
| OCR, URL input, Hindi/Hinglish | UNAVAILABLE (cut) | English only. The 17-clip Hindi demo set was not run |
| 30-60 s clip latency check (SPEC 13) | NOT TESTED | Only 2-8 s FakeAVCeleb clips were run: ~2.7-3.5 s per clip end to end (excl. one-off model load ~7 s warm) |

## Tests (real output)

`python -m pytest tests -q` -> **25 passed** (12 fusion with hand-computed numbers, 12 scam-rule, 1 integration test).
The integration test (`tests/test_pipeline_scam_path.py`) uses a MOCKED transcript on a real clip; it checks plumbing, not ASR quality.
No test was weakened. One change: `guard.window_tokens` was raised 6 -> 8 in `rules.yaml` because my own test
"Beware of callers who ask you to send your OTP." has the warning word 7 tokens before the match.

## Data

* DATASET_ROOT from `.env`: `C:\Users\Purva\Downloads\archive\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2` (the `.env` value overrides a shell variable, `load_dotenv(override=True)`).
* 21,560 `.mp4` on disk (RVRA 500, RVFA 500, FVRA 9709, FVFA 10851), all in the 4 category folders; no `frames/` or `moved/` folders exist.
* `meta_data.csv`: 21,566 rows. The header is shifted: column `path` holds the file name and the unnamed 10th column holds the directory. The loader detects this instead of assuming the schema. 22 duplicate rows dropped; 21,544 rows matched to files; 16 files had no row.
* Grouping: connected components over source/target1/target2 gave **1 group = 100% of clips -> assertion FAILED**. Regrouped by the `source` speaker column alone: 500 groups, largest 88 clips (0.4%), assertion passed. **Method used: source-speaker alone.**
* Split: whole groups 80/20 (train 400 groups / val 100 groups over all matched clips); zero overlap asserted.
* Subset: 40 clips per category (32 train + 8 val), max 2 clips per group; every clip verified as OpenCV-readable .mp4 with an ffmpeg audio stream.

| split | RVRA | RVFA | FVRA | FVFA | total |
|---|---|---|---|---|---|
| train | 32 | 32 | 32 | 32 | 128 |
| val | 8 | 8 | 8 | 8 | 32 |

Subset groups: train 89, val 21.

## Head metrics (val only, small subset, indicative)

Clip score = mean of top-3 windows (visual window score = 0.5*mean(p)+0.5*max(p) over the window's crops). Threshold 0.5 for P/R/F1.
Hyper-parameters (C; Wav2Vec2 layer) were chosen on these same val clips, so the numbers are optimistic.

| Head | rows train / val | chosen | clip ROC-AUC | precision | recall | F1 |
|---|---|---|---|---|---|---|
| Visual (label = video_label) | 434 / 92 crops | C=0.3 | **0.723** | 0.600 | 0.938 | 0.732 |
| Audio (label = audio_label) | 211 / 46 windows | layer 6, C=0.03 | **1.000** | 1.000 | 1.000 | 1.000 |

* Per-category AUC (category vs. all opposite-label val clips; 8 clips each): visual FVFA 0.742, FVRA 0.703, RVFA 0.734, RVRA 0.711; audio 1.000 in all four.
* Real val clips by band (held-out): visual LOW 6 / MEDIUM 3 / **HIGH 7** (of 16); audio LOW 15 / MEDIUM 1 / HIGH 0.
* Extra (train only): grouped 5-fold out-of-fold clip AUC visual 0.743, audio 1.000; real train clips by band visual 28/8/**28**, audio 61/3/0.
* Neither head is below 0.65, so the 0.3 low-confidence weight did not trigger (the mechanism is implemented and shown in the UI when it does).
* Shortcut check (LR on duration, loudness, fps, width, height only), val AUC: video_label **0.703**, audio_label **0.832**. All clips are 224x224 and ~25 fps, so only duration and loudness carry signal (fake clips are slightly shorter/quieter on average).
* Sanity check: audio head trained with shuffled clip labels gives val AUC 0.43-0.61 over 5 shuffles, so the 1.0 is not a pipeline bug.

## Deviations and caveats

* **Windows Application Control policy** blocked some DLLs of the newest wheels (scipy 1.18 `_interpnd`, scikit-learn 1.9 `_middle_term_computer`, av 19 `_core`). I installed older releases (scipy 1.17.1, scikit-learn 1.8.0, av 17.1.0; pinned in `requirements.txt`); I did not modify any policy.
* **Audio AUC 1.0 is dataset-specific.** FakeAVCeleb's cloned voices (SV2TTS) are easy to separate and the metadata shortcut AUC is 0.83, so expect much worse on real-world ads and other voice-cloning tools.
* **Visual head is weak** (0.72; 7/16 real val clips score HIGH). It is trained on only 128 clips / 434 crops with 2048-d features. Its alerts should be read together with R_v; the UI shows scores as /100*.
* **Identity leakage risk:** grouping is by source speaker only. Target identities (faceswap / voice target) can appear on both sides of the split.
* Val is 32 clips (16 per class): AUC differences of a few points are noise.
* Reliability formulas and thresholds (FACE_SIZE_FULL 0.15, BLUR_FULL 100, SNR_FULL 20 dB, etc.) are uncalibrated design choices; BLUR_FULL was set after looking at the pooled blur distribution of the extracted clips (median 185), not at labels.
* Evidence cards only list detector windows with score >= 0.5 (`EVIDENCE_MIN_SCORE`) so a clean clip is not shown "evidence" of fakeness.
* Scam engine was verified with unit tests and a mocked transcript only; there was no real scam-ad video, and Whisper-base on VoxCeleb speech produced rough transcripts (e.g. conf 0.42 on one real clip).
* Clips are short (2-8 s), so most have 1-3 windows; top-3/25% aggregation is mostly exercised in unit tests, not by real videos.
* `opencv-python` and `opencv-contrib-python` (pulled in by mediapipe) are both installed; cv2 5.0.0 works.
* Python files written by the editor tool came out as UTF-16; `fixenc.py` converts them to UTF-8 (helper kept in the repo root).
* Milestone times (Get-Date): M0 04:54-05:02 (overran: DLL block), M1 05:04-05:06, M2 05:07-05:12 (249 s extraction), M3 05:08-05:12, M4 05:12, M5 05:13-05:17.

---

## Update: 800-clip development set (supersedes the 160-clip head numbers above)

Commands: `python scripts/build_dev_manifest.py` -> `outputs/manifests/dev_800.csv`;
`python scripts/extract.py --manifest outputs/manifests/dev_800.csv --fail-log outputs/logs/extract_800_failures.log --workers 2`
(only the 640 new clips were extracted; the 160 cached ones were reused);
`python scripts/train_heads.py --manifest outputs/manifests/dev_800.csv`; `python scripts/compare_160_vs_800.py`.
Previous 160-clip heads/metrics are kept in `models/backup_160/`, `outputs/metrics_160.json`, `outputs/val_predictions_160.csv`.

* Composition: 800 clips = 200 per category (train 160 + val 40 each); train 640 / val 160; 373 source-speaker groups (train 299, val 74); train/val group overlap 0; max 6 clips per group.
  The original 160 clips kept the same clip_id, group and split (asserted by recomputing the split). Seed 42.
* Extraction: 640/640 new clips ok, 0 failures, 1136 s wall. Usable: 800 for the visual head, 798 for the audio head (c448 and c663 have no usable >= 1 s speech window).
* No frozen test set exists in this project (the brief had "no test split"); none was found or opened, so absence could only be asserted by "nothing defined", not by intersection.

| Head (val, own split) | 160-clip model (32 val clips) | 800-clip model (160 val clips) |
|---|---|---|
| Visual ROC-AUC | 0.723 | 0.899 (C=0.03) |
| Visual precision / recall / F1 @0.5 | 0.600 / 0.938 / 0.732 | 0.767 / 0.863 / 0.812 |
| Visual confusion [[TN,FP],[FN,TP]] | n/a (not logged) | [[59, 21], [11, 69]] |
| Audio ROC-AUC (layer 6, C=0.03) | 1.000 | 1.000 (159 val clips) |
| Audio P / R / F1 | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0, confusion [[80,0],[0,79]] |
| Shortcut LR val AUC (video / audio label) | 0.703 / 0.832 | 0.716 / 0.650 |

Like-for-like: the new heads on the ORIGINAL 32 val clips give visual AUC 0.926 (old model 0.723), audio 1.000.
Visual grouped OOF AUC on train 0.743 -> 0.790. Visual improved on every measured metric except recall; audio was already saturated, so no change is measurable.
Caveats unchanged: C and layer are picked on val (optimistic); the audio 1.000 is dataset-specific; the visual head still scores 9/80 real val clips HIGH and 25 MEDIUM.
`models/*.pkl` and `models/head_meta.json` now hold the 800-clip heads used by the pipeline, CLI and app. Tests: 25 passed.
