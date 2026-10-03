"""Streamlit UI: Deepfake Scam Ad Detector (English only).  Run: streamlit run app.py"""
import html
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from src import config, fusion
from src.pipeline import load_all, mmss, run_pipeline

st.set_page_config(page_title="Deepfake Scam Ad Detector", layout="wide")

BAND_STYLE = {
    "LOW": ("#1b7f3b", "LOW RISK"),
    "MEDIUM": ("#b7791f", "MEDIUM RISK"),
    "HIGH": ("#c0262d", "HIGH RISK"),
    fusion.INSUFFICIENT: ("#5b6470", "INSUFFICIENT EVIDENCE"),
}
CAT_COLORS = {"LOW": "#2e9e55", "MEDIUM": "#e0a030", "HIGH": "#d63a3a", fusion.INSUFFICIENT: "#8a929c"}
FOOTER = ("Scores are heuristic evidence scores, not probabilities. Results depend on video quality, face visibility, "
          "audio quality, compression, language and model coverage.")

STAGE_PROGRESS = [  # (substring in stage name, start fraction, span)
    ("Loading", 0.00, 0.05), ("Ingest", 0.05, 0.02), ("Detecting faces", 0.07, 0.10),
    ("Xception", 0.17, 0.10), ("Wav2Vec2", 0.27, 0.30), ("Facial landmarks", 0.57, 0.04), ("Scoring", 0.61, 0.02),
    ("Transcribing", 0.63, 0.22), ("Fusing", 0.85, 0.05), ("Building", 0.90, 0.07), ("Done", 1.0, 0.0),
]


@st.cache_resource(show_spinner="Loading frozen models (once)...")
def _models():
    return load_all()


def score_txt(x):
    return "N/A" if x is None else f"{round(100 * x)}/100*"


def rel_txt(x):
    return "N/A" if x is None else f"{x:.2f}"


def score_bar(label, x, note=""):
    st.markdown(f"**{label}**: {score_txt(x)}" + (f"  \n<small>{note}</small>" if note else ""), unsafe_allow_html=True)
    if x is None:
        st.markdown("<div style='background:#e8eaed;border-radius:6px;padding:4px 10px;color:#5b6470'>N/A - not evaluated</div>",
                    unsafe_allow_html=True)
    else:
        st.progress(min(1.0, max(0.0, float(x))))


def render_transcript(rep):
    if not rep["transcript_ok"] and not rep["transcript"]:
        st.info("Transcript N/A (no speech detected, no audio, or transcription failed).")
        return
    parts = []
    for s in rep["transcript"]:
        text = s["text"]
        marks = sorted(s["hits"], key=lambda h: h["start"])
        out, pos = "", 0
        for h in marks:
            if h["start"] < pos:
                continue
            out += html.escape(text[pos:h["start"]])
            span = html.escape(text[h["start"]:h["end"]])
            if h["suppressed"]:
                out += (f"<span style='background:#e8eaed;text-decoration:line-through' "
                        f"title='{html.escape(h['reason'])}'>{span}</span>")
            else:
                out += f"<mark style='background:#ffd54f' title='{html.escape(h['category'])}'>{span}</mark>"
            pos = h["end"]
        out += html.escape(text[pos:])
        parts.append(f"<div><code>{mmss(s['start'])}-{mmss(s['end'])}</code> {out} "
                     f"<small style='color:#777'>(ASR confidence {s['conf']:.2f})</small></div>")
    st.markdown("".join(parts), unsafe_allow_html=True)
    st.caption("Yellow = scam-rule match. Grey strike-through = matched but suppressed by the context guard "
               "(e.g. a warning such as 'never share your OTP').")


def render_timeline(rep):
    cells = []
    total = max(w["end"] for w in rep["windows"])
    for w in rep["windows"]:
        c = CAT_COLORS[w["band"]]
        tip = (f"{mmss(w['start'])}-{mmss(w['end'])} | {w['band']} | D {score_txt(w['D'])} | S {score_txt(w['S'])} | "
               f"R_v {rel_txt(w['R_v'])} R_a {rel_txt(w['R_a'])}")
        cells.append(f"<div title='{html.escape(tip)}' style='flex:1;min-width:34px;background:{c};color:white;"
                     f"padding:10px 4px;text-align:center;font-size:12px;border-right:2px solid white'>"
                     f"{mmss(w['start'])}<br>{w['band'][:4]}</div>")
    st.markdown(f"<div style='display:flex;border-radius:6px;overflow:hidden'>{''.join(cells)}</div>", unsafe_allow_html=True)
    st.caption("4 s windows, 2 s hop (windows overlap). Colour = per-window band; grey = insufficient evidence. Hover for details.")


def show_report(rep):
    v = rep["video"]
    color, label = BAND_STYLE[v["band"]]
    st.markdown(f"<div style='background:{color};color:white;padding:18px 22px;border-radius:10px;font-size:28px;"
                f"font-weight:700'>{label}<span style='font-size:15px;font-weight:400;margin-left:18px'>"
                f"overall evidence {score_txt(v['R_video'])}</span></div>", unsafe_allow_html=True)
    if v["band"] == fusion.INSUFFICIENT:
        st.warning("Not enough usable evidence (media unreliable and no scam language found). This is NOT a 'low risk' result.")
    st.write("")
    c1, c2, c3 = st.columns(3)
    with c1:
        note = ""
        lc = rep["low_confidence"]
        if lc["visual"] or lc["audio"]:
            note = "includes low-confidence " + " and ".join(
                n for n, f in (("visual analysis", lc["visual"]), ("voice analysis", lc["audio"])) if f)
        score_bar("Deepfake evidence (D)", v["D"], note)
    with c2:
        score_bar("Scam evidence (S)", v["S"])
    with c3:
        sy = v["sync"]
        score_bar("AV mismatch (M)", None, "Not fused (supporting sync score shown below)")
        st.caption("AV sync (supporting): " + (f"correlation {sy['sync_score']:.2f}, lag {sy['best_lag_ms']:.0f} ms, "
                   f"reliability {sy['reliability']:.2f}, {sy['n_valid_windows']} window(s)" if sy["sync_score"] is not None
                   else f"N/A - {sy['reason']}"))
    st.caption(f"Media / deepfake risk: **{v['media_band']}** | Scam risk: **{v['scam_band']}** (reported separately; the banner above is the SPEC fusion) "
               f"| windows with D >= 0.5: {v['coverage']['n_windows_D_ge_0.5']}/{v['coverage']['n_windows_with_D']}")
    st.caption("* heuristic evidence score out of 100, not a probability.")

    st.subheader("Reliability")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("R_v (face)", rel_txt(v["R_v"]))
    r2.metric("R_a (audio)", rel_txt(v["R_a"]))
    r3.metric("R_t (transcript)", rel_txt(v["R_t"]))
    r4.metric("R_sum (media)", f"{v['R_sum']:.2f}")
    st.caption(f"A modality below {config.R_GATE} is dropped and shown as N/A, never as 0. "
               f"Face backend: {rep['face_backend']}.")
    for w in rep["warnings"]:
        st.warning(w)

    st.subheader("Window timeline")
    render_timeline(rep)

    st.subheader("Top evidence")
    if not rep["evidence"]:
        st.info(f"No window scored >= {config.EVIDENCE_MIN_SCORE} on any detector and no scam rule matched.")
    for i, e in enumerate(rep["evidence"], 1):
        with st.container(border=True):
            cols = st.columns([1, 4])
            if e["thumbnail_path"]:
                cols[0].image(e["thumbnail_path"], width=130)
            else:
                cols[0].caption("no thumbnail")
            cols[1].markdown(f"**#{i} {e['modality'].upper()}** | {mmss(e['t_start'])}-{mmss(e['t_end'])} | "
                             f"score {score_txt(e['score'])} | reliability {e['reliability']:.2f}")
            cols[1].write(e["reason"])
            if e["text"]:
                cols[1].caption(f"Transcript: \"{e['text']}\"")

    st.subheader("Scam checklist (6 categories)")
    cc = st.columns(3)
    for i, c in enumerate(rep["checklist"]):
        with cc[i % 3]:
            st.markdown(f"{'🔴' if c['found'] else '⚪'} **{c['label']}** - {'matched' if c['found'] else 'not found'}")
            for m in c["matches"][:3]:
                st.caption(f"\"{m['text']}\" at {mmss(m['t'])}")
    if not rep["transcript_ok"]:
        st.caption("Transcript unavailable: the checklist is N/A, not 'clean'.")

    st.subheader("Transcript (English only)")
    render_transcript(rep)

    with st.expander("Per-stage timings and model info"):
        st.json({k: round(t, 2) for k, t in rep["timings"].items()})
        st.json({"head_meta": rep["head_meta"], "fusion_weights": rep["weights"]})
    with st.expander("Per-window scores"):
        st.dataframe([{"window": f"{mmss(w['start'])}-{mmss(w['end'])}", "band": w["band"],
                       "S_visual": score_txt(w["S_visual"]), "R_v": rel_txt(w["R_v"]),
                       "S_audio": score_txt(w["S_audio"]), "R_a": rel_txt(w["R_a"]),
                       "S_scam": score_txt(w["S"]), "R_t": rel_txt(w["R_t"]),
                       "D": score_txt(w["D"]), "faces": f"{w['n_faces']}/{w['n_frames']}",
                       "sync": ("N/A" if w["sync"]["sync_score"] is None else f"{w['sync']['sync_score']:.2f} @ {w['sync']['best_lag_ms']:.0f}ms"),
                       "blinks": ("N/A" if not w["face_temporal"]["available"] else int(w["face_temporal"]["features"]["blink_count"])),
                       "MAR std": ("N/A" if not w["face_temporal"]["available"] else round(w["face_temporal"]["features"]["mar_std"], 3)),
                       "pitch Hz": ("N/A" if w["acoustic"]["features"].get("pitch_mean") is None else round(w["acoustic"]["features"]["pitch_mean"]))}
                      for w in rep["windows"]])


def main():
    st.title("Deepfake Scam Ad Detector")
    up = st.file_uploader("Upload a video ad (mp4 / mov / webm)", type=["mp4", "mov", "webm"])
    if up is not None:
        key = f"{up.name}-{up.size}"
        if st.session_state.get("key") != key:
            _models()
            suffix = Path(up.name).suffix or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(up.getbuffer())
            bar, status = st.progress(0.0), st.empty()

            def cb(stage, frac=0.0):
                for name, start, span in STAGE_PROGRESS:
                    if name in stage:
                        bar.progress(min(1.0, start + span * float(frac)))
                        break
                status.write(f"Stage: {stage} ...")

            try:
                st.session_state["report"] = run_pipeline(tmp.name, cb)
                st.session_state["key"] = key
                st.session_state["error"] = None
            except Exception as e:  # noqa: BLE001
                st.session_state["report"] = None
                st.session_state["key"] = key
                st.session_state["error"] = f"{type(e).__name__}: {e}"
            bar.empty()
            status.empty()
        if st.session_state.get("error"):
            st.error(f"Analysis could not be completed: {st.session_state['error']}")
        elif st.session_state.get("report"):
            show_report(st.session_state["report"])
    st.divider()
    st.caption(FOOTER)


if __name__ == "__main__":
    main()
