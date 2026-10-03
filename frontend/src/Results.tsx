import { Component, useRef, useState, type ReactNode } from "react";
import {
  Activity,
  AudioLines,
  Check,
  ChevronDown,
  ChevronRight,
  Clock3,
  Download,
  Eye,
  Mic2,
  ScanFace,
  ShieldCheck,
  TriangleAlert,
} from "lucide-react";
import type { AnalysisResult, Band, EvidenceItem, ModalityBlock, TranscriptSegment } from "./api";

export type ResultVideo = { name: string; url: string | null };

/* ---------- formatting helpers (display only; no scoring happens in the browser) ---------- */

const mmss = (t: number) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
const span = (a: number, b: number) => `${mmss(a)}–${mmss(b)}`;
const score100 = (x: number | null) => (x === null || x === undefined ? null : Math.round(100 * x));
const num = (x: number | null | undefined, d = 2) => (x === null || x === undefined ? "N/A" : x.toFixed(d));

type Tone = "low" | "medium" | "high" | "na";
const bandTone = (b: Band): Tone => (b === "LOW" ? "low" : b === "MEDIUM" ? "medium" : b === "HIGH" ? "high" : "na");

// Reliability is shown as a plain-language level. The 0.7 / 0.4 cut-offs are display buckets only.
function relLevel(x: number | null | undefined): { label: string; tone: Tone } {
  if (x === null || x === undefined) return { label: "N/A", tone: "na" };
  if (x >= 0.7) return { label: "HIGH", tone: "low" };
  if (x >= 0.4) return { label: "MEDIUM", tone: "medium" };
  return { label: "LOW", tone: "high" };
}

function Chip({ tone = "na", children, title }: { tone?: Tone; children: ReactNode; title?: string }) {
  return <span className={`chip tone-${tone}`} title={title}>{children}</span>;
}

function Panel({ title, sub, right, children }: { title: string; sub?: string; right?: ReactNode; children: ReactNode }) {
  return (
    <section className="evidence-panel">
      <div className="evidence-heading">
        <div><span>{title}</span>{sub && <p>{sub}</p>}</div>
        {right}
      </div>
      {children}
    </section>
  );
}

/* ---------- risk blocks ---------- */

function RiskBlock({ label, icon, score, band, reason, note }: {
  label: string; icon: ReactNode; score: number | null; band: Band; reason: string | null; note?: string;
}) {
  const s = score100(score);
  const tone = bandTone(band);
  return (
    <div className="risk-block">
      <div className="flex items-center justify-between">
        <span className="flex items-center gap-2 text-[9px] font-medium uppercase tracking-[0.16em] text-slate-500">{icon}{label}</span>
        <Chip tone={tone}>{band === "N/A" ? "N/A" : `${band} risk`}</Chip>
      </div>
      {s === null ? (
        <>
          <div className="mt-4 text-4xl font-semibold tracking-[-0.04em] text-slate-500">N/A</div>
          <p className="mt-1 text-[11px] text-slate-500">{reason ?? "Not enough reliable evidence"}</p>
          <div className="meter"><i className="tone-na" style={{ width: 0 }} /></div>
        </>
      ) : (
        <>
          <div className="mt-4 text-5xl font-semibold tracking-[-0.05em] text-white">
            {s}<span className="text-xl text-slate-500">/100*</span>
          </div>
          <div className="meter"><i className={`tone-${tone}`} style={{ width: `${Math.max(2, s)}%` }} /></div>
        </>
      )}
      {note && <p className="mt-3 text-[10px] leading-4 text-slate-500">{note}</p>}
    </div>
  );
}

/* ---------- modality breakdown ---------- */

function ModalityRow({ icon, label, block }: { icon: ReactNode; label: string; block: Pick<ModalityBlock, "score" | "reliability" | "reason"> & { low_confidence?: boolean } }) {
  const s = score100(block.score);
  const rel = relLevel(block.reliability);
  return (
    <div className="modality-row">
      <div className="flex items-center gap-3">
        <div className="modality-icon">{icon}</div>
        <div className="min-w-0">
          <div className="text-[11px] font-medium text-slate-200">{label}</div>
          <div className="text-[9px] text-slate-500">
            {s === null ? (block.reason ?? "N/A") : block.low_confidence ? "Low-confidence analysis" : "Evidence score"}
          </div>
        </div>
      </div>
      <div className="flex items-center gap-3">
        <strong className={`text-sm ${s === null ? "text-slate-500" : "text-white"}`}>{s === null ? "N/A" : `${s}/100*`}</strong>
        <Chip tone={rel.tone} title={block.reliability === null ? undefined : `Reliability ${num(block.reliability)}`}>
          {rel.label === "N/A" ? "Reliability N/A" : `${rel.label} reliability`}
        </Chip>
      </div>
    </div>
  );
}

/* ---------- timeline ---------- */

function Timeline({ r }: { r: AnalysisResult }) {
  const rows: { key: "media" | "scam"; label: string }[] = [{ key: "media", label: "Media / deepfake" }, { key: "scam", label: "Scam language" }];
  return (
    <div className="px-4 pb-4 pt-4">
      {rows.map((row) => (
        <div key={row.key} className="mb-3">
          <div className="mb-1.5 text-[9px] uppercase tracking-[0.12em] text-slate-500">{row.label}</div>
          <div className="timeline-strip">
            {r.timeline.map((w, i) => {
              const d = w[row.key];
              const s = score100(d.score);
              return (
                <div key={i} className="timeline-cell">
                  <div
                    className={`tl-seg tone-${bandTone(d.band)}`}
                    title={`${span(w.start, w.end)} · ${d.band === "N/A" ? "N/A" : `${d.band} · ${s}/100*`}`}
                  >
                    {d.band === "N/A" ? "N/A" : s}
                  </div>
                  <span>{mmss(w.start)}</span>
                </div>
              );
            })}
          </div>
        </div>
      ))}
      <p className="text-[9px] text-slate-600">
        {r.video.window_s}-second windows, {r.video.hop_s}-second hop (neighbouring windows overlap). Hover for details; N/A means that signal could not be assessed in that window.
      </p>
    </div>
  );
}

/* ---------- evidence ---------- */

const MODALITY_LABEL: Record<EvidenceItem["modality"], string> = { visual: "Visual", audio: "Audio", scam: "Scam" };

function EvidenceRow({ e, onJump }: { e: EvidenceItem; onJump?: (t: number) => void }) {
  const rel = relLevel(e.reliability);
  const Wrapper = onJump ? "button" : "div";
  return (
    <Wrapper className="evidence-row items-start" {...(onJump ? { onClick: () => onJump(e.start) } : {})}>
      {e.thumbnail ? <img src={e.thumbnail} alt="" className="ev-thumb" /> : <div className="ev-thumb ev-thumb-empty"><ScanFace size={16} /></div>}
      <div className="min-w-0 flex-1 text-left">
        <div className="flex flex-wrap items-center gap-2">
          <span className="evidence-time"><Clock3 size={11} /> {span(e.start, e.end)}</span>
          <Chip tone="na">{MODALITY_LABEL[e.modality]}</Chip>
          <Chip tone={rel.tone} title={`Reliability ${num(e.reliability)}`}>{rel.label} reliability</Chip>
        </div>
        <p className="mt-1.5 text-[11px] leading-5 text-slate-300">{e.reason}</p>
        {e.text && <p className="mt-1 text-[10px] italic leading-4 text-slate-500">“{e.text}”</p>}
      </div>
      <strong>{Math.round(100 * e.score)}/100*</strong>
      {onJump && <ChevronRight size={13} />}
    </Wrapper>
  );
}

/* ---------- transcript ---------- */

function SegmentText({ s }: { s: TranscriptSegment }) {
  const parts: ReactNode[] = [];
  let pos = 0;
  [...s.matches].sort((a, b) => a.start - b.start).forEach((m, i) => {
    if (m.start < pos) return;
    parts.push(s.text.slice(pos, m.start));
    const piece = s.text.slice(m.start, m.end);
    parts.push(
      m.suppressed
        ? <span key={i} className="match-suppressed" title={m.note ?? "Matched but treated as a warning, not a scam claim"}>{piece}</span>
        : <mark key={i} title={m.label}>{piece}</mark>,
    );
    pos = m.end;
  });
  parts.push(s.text.slice(pos));
  return <>{parts}</>;
}

/* ---------- detailed analysis ---------- */

const ACOUSTIC_LABEL: Record<string, string> = {
  pitch_mean: "Voice pitch, mean (Hz)", pitch_std: "Voice pitch, variation (Hz)", rms_mean: "Loudness, mean", rms_std: "Loudness, variation",
  zcr_mean: "Zero-crossing rate", spectral_centroid_mean: "Spectral centroid (Hz)", speech_ratio: "Speech ratio", silence_ratio: "Silence ratio",
};
const FACE_LABEL: Record<string, string> = {
  mar_mean: "Mouth opening, mean", mar_std: "Mouth opening, variation", mar_range: "Mouth opening, range", mar_velocity: "Mouth movement speed",
  ear_mean: "Eye openness, mean", ear_std: "Eye openness, variation", blink_count: "Blinks (average per window)",
  yaw_mean: "Head yaw, mean", yaw_std: "Head yaw, variation", pitch_mean: "Head pitch, mean", pitch_std: "Head pitch, variation",
  roll_mean: "Head roll, mean", roll_std: "Head roll, variation",
};

function FeatureTable({ title, labels, block }: { title: string; labels: Record<string, string>; block: { available: boolean; reason: string | null; features: Record<string, number | null>; windows_available: number; windows_total: number } }) {
  return (
    <div className="detail-card">
      <div className="detail-title">{title}</div>
      {!block.available ? (
        <p className="text-[10px] text-slate-500">N/A{block.reason ? `: ${block.reason}` : ""}</p>
      ) : (
        <>
          <p className="mb-2 text-[9px] text-slate-600">Average over {block.windows_available} of {block.windows_total} windows</p>
          <dl className="detail-list">
            {Object.entries(block.features).map(([k, v]) => (
              <div key={k}><dt>{labels[k] ?? k}</dt><dd>{v === null ? "N/A" : Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2)}</dd></div>
            ))}
          </dl>
        </>
      )}
    </div>
  );
}

function DetailedAnalysis({ r }: { r: AnalysisResult }) {
  const [open, setOpen] = useState(false);
  return (
    <section className="evidence-panel mt-3.5">
      <button className="evidence-heading w-full text-left" onClick={() => setOpen(!open)} aria-expanded={open}>
        <div><span>Detailed analysis</span><p>Per-window scores and supporting measurements</p></div>
        <ChevronDown size={15} className={`text-slate-500 transition-transform ${open ? "rotate-180" : ""}`} />
      </button>
      {open && (
        <div className="p-4">
          <div className="overflow-x-auto">
            <table className="detail-table">
              <thead>
                <tr><th>Window</th><th>Visual</th><th>Audio</th><th>Media</th><th>Scam</th><th>Faces</th><th>AV sync*</th></tr>
              </thead>
              <tbody>
                {r.windows.map((w, i) => (
                  <tr key={i}>
                    <td>{span(w.start, w.end)}</td>
                    <td>{w.visual.score === null ? "N/A" : `${score100(w.visual.score)} · r${num(w.visual.reliability, 1)}`}</td>
                    <td>{w.audio.score === null ? "N/A" : `${score100(w.audio.score)} · r${num(w.audio.reliability, 1)}`}</td>
                    <td>{w.media.score === null ? "N/A" : score100(w.media.score)}</td>
                    <td>{w.scam.score === null ? "N/A" : score100(w.scam.score)}</td>
                    <td>{w.faces.frames_with_face}/{w.faces.frames}</td>
                    <td>{w.av_sync.available ? `${num(w.av_sync.score)} @ ${Math.round(w.av_sync.best_lag_ms ?? 0)} ms` : "N/A"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-2 text-[9px] text-slate-600">Scores are out of 100 (heuristic, not probabilities). “r” is the reliability of that signal (0 to 1). *AV sync is supporting evidence only.</p>
          <div className="mt-4 grid gap-3 md:grid-cols-2">
            <FeatureTable title="Voice measurements (supporting)" labels={ACOUSTIC_LABEL} block={r.acoustic} />
            <FeatureTable title="Facial movement measurements (supporting)" labels={FACE_LABEL} block={r.facial_temporal} />
          </div>
          <p className="mt-3 text-[9px] leading-4 text-slate-600">
            Voice and facial-movement measurements are supporting information. They are not part of the trained deepfake detectors or of the scores above.
          </p>
          {r.technical_notes.length > 0 && (
            <ul className="mt-3 list-disc space-y-1 pl-4 text-[10px] text-slate-500">
              {r.technical_notes.map((n, i) => <li key={i}>{n}</li>)}
            </ul>
          )}
        </div>
      )}
    </section>
  );
}

/* ---------- page ---------- */

function summaryText(r: AnalysisResult, name: string): string {
  const s = (x: number | null) => (x === null ? "N/A" : `${Math.round(100 * x)}/100*`);
  return [
    "VERISIGHT AI — VIDEO VERIFICATION SUMMARY",
    "",
    `File: ${name}`,
    `Analysis ID: ${r.analysis_id}`,
    "",
    `Media / deepfake risk: ${s(r.media_risk.score)} (${r.media_risk.band})`,
    `Scam risk: ${s(r.scam_risk.score)} (${r.scam_risk.band})`,
    r.insufficient_evidence ? "Note: not enough usable evidence for a confident overall result." : "",
    "",
    `Visual evidence: ${s(r.visual.score)}${r.visual.available ? "" : ` (${r.visual.reason})`}`,
    `Audio evidence: ${s(r.audio.score)}${r.audio.available ? "" : ` (${r.audio.reason})`}`,
    `Language evidence: ${s(r.semantic.score)}${r.semantic.available ? "" : ` (${r.semantic.reason})`}`,
    `AV sync (supporting only): ${r.av_sync.available ? `correlation ${num(r.av_sync.score)}, lag ${Math.round(r.av_sync.best_lag_ms ?? 0)} ms` : `N/A (${r.av_sync.reason})`}`,
    "",
    "EVIDENCE",
    ...(r.evidence.length ? r.evidence.map((e) => `${span(e.start, e.end)} [${MODALITY_LABEL[e.modality]}] ${e.reason}`) : ["No suspicious moments found."]),
    "",
    "* Heuristic evidence score out of 100, not a probability. Results depend on video quality, face visibility, audio quality, compression, language and model coverage.",
  ].filter((l, i, a) => l !== "" || a[i - 1] !== "").join("\n");
}

function ResultsView({ result: r, video, onAgain }: { result: AnalysisResult; video: ResultVideo | null; onAgain: () => void }) {
  console.log("RENDERING ANALYSIS RESULT:", r);
  const videoRef = useRef<HTMLVideoElement>(null);
  const jumpTo = (seconds: number) => {
    if (!videoRef.current) return;
    videoRef.current.currentTime = seconds;
    void videoRef.current.play();
  };
  const downloadSummary = () => {
    const url = URL.createObjectURL(new Blob([summaryText(r, video?.name ?? "saved analysis")], { type: "text/plain" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = "verisight-verification-summary.txt";
    a.click();
    URL.revokeObjectURL(url);
  };

  const m = r.media_risk;
  const top = r.evidence[0];
  const reliabilityChips: { label: string; value: number | null | undefined; available: boolean }[] = [
    { label: "Visual", value: r.reliability.visual, available: r.visual.available },
    { label: "Audio", value: r.reliability.audio, available: r.audio.available },
    { label: "Text", value: r.reliability.transcript, available: r.semantic.available },
    { label: "Sync", value: r.av_sync.reliability, available: r.av_sync.available },
  ];
  const detected = r.semantic.categories.filter((c) => c.found).length;

  return (
    <main className="page-wrap">
      <div className="mx-auto w-full max-w-5xl">
        <div className="mb-7 flex flex-col justify-between gap-4 sm:flex-row sm:items-end">
          <div>
            <div className="section-label"><span className="!bg-emerald-400" /> Analysis complete</div>
            <h1 className="mt-3 text-3xl font-semibold tracking-tight text-white">Verification result</h1>
            <p className="mt-2 text-xs text-slate-600">{video?.name ?? "Uploaded video"}</p>
          </div>
          <div className="flex flex-wrap gap-2">
            <button className="outline-button" onClick={downloadSummary}><Download size={13} /> Download summary</button>
            <button className="outline-button" onClick={onAgain}>Analyze another <ChevronRight size={14} /></button>
          </div>
        </div>

        <div className="results-grid">
          <div className="video-frame min-h-[320px]">
            {video?.url ? (
              <video ref={videoRef} src={video.url} className="size-full object-contain" controls />
            ) : (
              <p className="relative z-10 px-6 text-center text-[11px] text-slate-500">Video preview is not available for a saved analysis.</p>
            )}
            <span className="corner corner-tl" /><span className="corner corner-tr" /><span className="corner corner-bl" /><span className="corner corner-br" />
          </div>

          <div className="result-panel">
            <RiskBlock
              label="Media / deepfake risk" icon={<Eye size={13} />} score={m.score} band={m.band} reason={m.reason}
              note={m.score !== null ? `${m.windows_flagged} of ${m.windows_scored} segments scored 50 or higher on face and voice evidence.` : undefined}
            />
            <div className="my-5 border-t border-white/[0.06]" />
            <RiskBlock label="Scam risk" icon={<Activity size={13} />} score={r.scam_risk.score} band={r.scam_risk.band} reason={r.scam_risk.reason}
              note={r.scam_risk.score !== null ? `${detected} of ${r.semantic.categories.length} scam signal types found in the speech.` : undefined} />
            {r.insufficient_evidence && (
              <div className="notice mt-5"><TriangleAlert size={13} /> Not enough usable evidence for a confident overall result. This is not a “low risk” result.</div>
            )}
            <div className="signal-summary">
              <div><Check size={13} /><span><strong>{r.evidence.length}</strong> evidence moment{r.evidence.length === 1 ? "" : "s"}</span></div>
              <div><TriangleAlert size={13} /><span><strong>{detected}</strong> scam signal type{detected === 1 ? "" : "s"}</span></div>
            </div>
          </div>
        </div>

        <div className="verification-strip">
          <div><span>Analysis ID</span><strong>{r.analysis_id}</strong></div>
          <div><span>Length analysed</span><strong>{mmss(r.video.analysed_s)}{r.video.truncated ? ` of ${mmss(r.video.duration_s ?? 0)}` : ""}</strong></div>
          <div><span>Segments</span><strong>{r.video.windows} × {r.video.window_s}s</strong></div>
          <div><span>Audio track</span><strong className={r.video.has_audio ? "" : "!text-amber-300"}>{r.video.has_audio ? "Present" : "None"}</strong></div>
        </div>

        {top && (
          <div className="notice notice-info mt-3.5">
            <Eye size={13} /> <span><strong>Top finding ({span(top.start, top.end)}):</strong> {top.reason}</span>
          </div>
        )}

        <div className="evidence-grid">
          <Panel title="Why it was flagged" sub="Most informative moments, ranked by score and reliability"
            right={<span className="evidence-count">{r.evidence.length} {r.evidence.length === 1 ? "moment" : "moments"}</span>}>
            {r.evidence.length === 0 ? (
              <p className="px-4 py-5 text-[11px] text-slate-500">No suspicious moments found.</p>
            ) : (
              r.evidence.map((e) => <EvidenceRow key={e.rank} e={e} onJump={video?.url ? jumpTo : undefined} />)
            )}
          </Panel>

          <Panel title="Reliability" sub="How much each signal can be trusted"
            right={<ShieldCheck size={15} className="text-blue-400" />}>
            <div className="p-4">
              <div className="flex flex-wrap gap-2">
                {reliabilityChips.map((c) => {
                  const lvl = c.available ? relLevel(c.value) : { label: "N/A", tone: "na" as Tone };
                  return <Chip key={c.label} tone={lvl.tone} title={c.value == null ? undefined : `Reliability ${num(c.value)}`}>{c.label}: {lvl.label}</Chip>;
                })}
              </div>
              {r.warnings.length > 0 && (
                <ul className="mt-4 list-disc space-y-1.5 pl-4 text-[11px] leading-5 text-slate-400">
                  {r.warnings.map((w, i) => <li key={i}>{w}</li>)}
                </ul>
              )}
              <p className="mt-4 text-[9px] leading-4 text-slate-600">
                A signal below the reliability threshold is left out and shown as N/A. It is never counted as zero.
              </p>
            </div>
          </Panel>
        </div>

        <div className="evidence-grid" style={{ marginTop: 14 }}>
          <Panel title="Evidence by signal" sub="Each signal is scored separately">
            <ModalityRow icon={<ScanFace size={14} />} label="Visual (face appearance)" block={r.visual} />
            <ModalityRow icon={<AudioLines size={14} />} label="Audio (voice)" block={r.audio} />
            <ModalityRow icon={<Mic2 size={14} />} label="Language (scam wording)" block={r.semantic} />
          </Panel>

          <Panel title="AV sync" sub="Supporting evidence only"
            right={<span className="evidence-count">Not in the scores</span>}>
            <div className="p-4">
              {r.av_sync.available ? (
                <>
                  <div className="flex items-end gap-3">
                    <div className="text-3xl font-semibold tracking-tight text-white">{num(r.av_sync.score)}</div>
                    <div className="mb-1 text-[10px] text-slate-500">mouth–audio correlation · lag {Math.round(r.av_sync.best_lag_ms ?? 0)} ms</div>
                  </div>
                  <p className="mt-2 text-[9px] text-slate-600">Reliability {num(r.av_sync.reliability)} · {r.av_sync.windows_used} window{r.av_sync.windows_used === 1 ? "" : "s"}</p>
                </>
              ) : (
                <>
                  <div className="text-3xl font-semibold tracking-tight text-slate-500">N/A</div>
                  <p className="mt-1 text-[11px] text-slate-500">{r.av_sync.reason ?? "Not analysed"}</p>
                </>
              )}
              <p className="mt-4 text-[10px] leading-4 text-slate-500">
                A lightweight check of whether mouth movement and audio energy move together. It is not a trained deepfake detector, and dubbed or voice-over videos can legitimately be out of sync.
              </p>
            </div>
          </Panel>
        </div>

        <div style={{ marginTop: 14 }}>
          <Panel title="Timeline" sub="Segment-by-segment view of the video">
            <Timeline r={r} />
          </Panel>
        </div>

        <div className="evidence-grid" style={{ marginTop: 14 }}>
          <Panel title="Transcript" sub="Speech heard in the video (English only)"
            right={<Mic2 size={15} className="text-violet-400" />}>
            {!r.transcript_available ? (
              <p className="px-4 py-5 text-[11px] text-slate-500">{r.semantic.reason ?? "No transcript available."}</p>
            ) : (
              <div className="transcript-scroll">
                {r.transcript_segments.map((s, i) => (
                  <div key={i} className="transcript-line">
                    <span className="evidence-time">{mmss(s.start)}</span>
                    <p><SegmentText s={s} /></p>
                  </div>
                ))}
              </div>
            )}
            {r.transcript_available && !r.scam_analysis_available && (
              <p className="px-4 pb-4 text-[10px] text-slate-500">Speech was too unclear to check for scam language.</p>
            )}
          </Panel>

          <Panel title="Scam signals" sub="Six common scam patterns">
            <div className="flex flex-wrap gap-2 p-4">
              {r.semantic.categories.map((c) => (
                <div key={c.id} className={`cat-pill ${c.found ? "cat-pill-on" : ""}`} title={c.matches.map((x) => `“${x.text}” at ${mmss(x.at)}`).join("\n")}>
                  <span>{c.label}</span>
                  {c.found && <small>“{c.matches[0].text}”{c.matches.length > 1 ? ` +${c.matches.length - 1}` : ""}</small>}
                </div>
              ))}
            </div>
            <p className="px-4 pb-4 text-[10px] text-slate-600">
              {r.semantic.categories_available ? "Highlighted = found in the speech. Grey = not found." : "Not analysed: no reliable speech, so these are N/A rather than clear."}
            </p>
          </Panel>
        </div>

        <DetailedAnalysis r={r} />

        <div className="mt-4 flex items-start gap-3 rounded-lg border border-blue-400/10 bg-blue-400/[0.035] px-4 py-3">
          <ShieldCheck size={15} className="mt-0.5 shrink-0 text-blue-400" />
          <p className="text-[10px] leading-5 text-slate-500">
            *Scores are heuristic evidence scores, not probabilities. Results depend on video quality, face visibility, audio quality, compression, language and model coverage.
          </p>
        </div>
      </div>
    </main>
  );
}

/* A rendering bug in the results must never blank the whole app. */
class ResultsBoundary extends Component<{ children: ReactNode; onAgain: () => void }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  render() {
    if (!this.state.failed) return this.props.children;
    return (
      <main className="page-wrap">
        <div className="mx-auto w-full max-w-xl text-center">
          <div className="notice justify-center"><TriangleAlert size={14} /> The result could not be displayed.</div>
          <button className="outline-button mt-4" onClick={this.props.onAgain}>Analyze another</button>
        </div>
      </main>
    );
  }
}

export default function ResultsPage(props: { result: AnalysisResult; video: ResultVideo | null; onAgain: () => void }) {
  return <ResultsBoundary onAgain={props.onAgain}><ResultsView {...props} /></ResultsBoundary>;
}
