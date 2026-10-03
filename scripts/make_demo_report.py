"""Run the REAL pipeline once on a local video and save the unmodified report for the UI's demo mode.

    python scripts/make_demo_report.py data/real_world_eval/synthetic_tts_scam.mp4

Writes outputs/reports/demo_report.json (used by `streamlit run app.py` + `?demo=1`).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.pipeline import run_pipeline  # noqa: E402


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    video = Path(sys.argv[1])
    out = config.OUT_DIR / "reports" / "demo_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    rep = run_pipeline(str(video), lambda stage, frac=0.0: print(f"[{frac:4.0%}] {stage}", flush=True))
    out.write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
