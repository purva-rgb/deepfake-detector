import { useEffect, useRef, useState, type DragEvent } from "react";
import {
  Activity,
  ArrowRight,
  Check,
  Eye,
  FileVideo2,
  Fingerprint,
  Mic2,
  Play,
  ScanLine,
  ShieldCheck,
  Sparkles,
  Sun,
  TriangleAlert,
  Upload,
  Video,
  X,
} from "lucide-react";
import { analyzeVideo, ApiError, fetchProgress, type AnalysisResult } from "./api";
import ResultsPage from "./Results";

type Page = "home" | "upload" | "results";

type SelectedVideo = {
  name: string;
  size: string;
  url: string;
  file: File;
};

const ACCEPTED_EXT = [".mp4", ".mov", ".webm"];

/* Pipeline stage names (from the backend progress callback) -> plain-language label and step index. */
const STAGES: { match: string; label: string; step: number }[] = [
  { match: "Uploading", label: "Uploading video", step: 0 },
  { match: "Loading", label: "Preparing analysis", step: 0 },
  { match: "Ingest", label: "Reading video and audio", step: 0 },
  { match: "Detecting faces", label: "Finding faces", step: 1 },
  { match: "Xception", label: "Analysing face appearance", step: 1 },
  { match: "Wav2Vec2", label: "Analysing voice", step: 1 },
  { match: "Facial landmarks", label: "Tracking facial movement", step: 1 },
  { match: "Scoring", label: "Scoring visual and audio evidence", step: 1 },
  { match: "Transcribing", label: "Transcribing speech", step: 2 },
  { match: "Fusing", label: "Combining evidence", step: 3 },
  { match: "Building", label: "Building the evidence list", step: 3 },
  { match: "Done", label: "Finishing up", step: 3 },
];

function Brand() {
  return (
    <div className="flex items-center gap-3">
      <div className="brand-mark">
        <Eye size={20} />
        <span className="brand-scan" />
      </div>
      <div>
        <div className="text-[13px] font-semibold tracking-[0.17em] text-white">VERISIGHT</div>
        <div className="text-[8px] font-medium uppercase tracking-[0.28em] text-blue-400">AI verification</div>
      </div>
    </div>
  );
}

function Header({ page, onNavigate }: { page: Page; onNavigate: (page: Page) => void }) {
  const goToSection = (id: string) => {
    onNavigate("home");
    window.setTimeout(() => document.getElementById(id)?.scrollIntoView({ behavior: "smooth" }), 0);
  };
  return (
    <header className="site-header">
      <button onClick={() => onNavigate("home")}><Brand /></button>
      <nav className="header-nav">
        <button onClick={() => onNavigate("home")} className={`nav-link ${page === "home" ? "nav-link-active" : ""}`}>Home</button>
        <button onClick={() => goToSection("how-it-works")} className="nav-link">How It Works</button>
        <button onClick={() => goToSection("about")} className="nav-link">About</button>
      </nav>
      <button className="theme-button"><Sun size={14} /> Light mode</button>
    </header>
  );
}

function Dropzone({ onSelect, compact = false }: { onSelect: (file: File) => void; compact?: boolean }) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    const file = event.dataTransfer.files[0];
    if (file) onSelect(file);
  };

  return (
    <div
      className={`cyber-dropzone ${compact ? "min-h-[258px]" : "min-h-[330px]"} ${dragging ? "cyber-dropzone-active" : ""}`}
      onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
    >
      <input ref={inputRef} type="file" accept=".mp4,.mov,.webm,video/mp4,video/quicktime,video/webm" className="hidden" onChange={(event) => event.target.files?.[0] && onSelect(event.target.files[0])} />
      <div className="upload-orbit">
        <FileVideo2 size={21} />
        <span />
      </div>
      <h3 className="mt-5 text-sm font-semibold text-white">Drop your video here</h3>
      <p className="mt-2 text-[11px] text-slate-500">or browse securely from your device</p>
      <p className="mt-5 text-[9px] uppercase tracking-[0.14em] text-slate-700">MP4 · MOV · WEBM · up to 90 s</p>
      <button className="outline-button mt-5" onClick={() => inputRef.current?.click()}>
        <Upload size={14} /> Choose Video
      </button>
    </div>
  );
}

function VerificationCore() {
  return (
    <div className="verification-core" aria-hidden="true">
      <div className="core-ambient" />
      <span className="core-particle core-particle-1" />
      <span className="core-particle core-particle-2" />
      <span className="core-particle core-particle-3" />
      <div className="core-scene">
        <div className="silver-orbit silver-orbit-back" />
        <div className="diamond-cube">
          <div className="cube-face cube-face-front"><Eye size={26} /></div>
          <div className="cube-face cube-face-back" />
          <div className="cube-face cube-face-right" />
          <div className="cube-face cube-face-left" />
          <div className="cube-face cube-face-top" />
          <div className="cube-face cube-face-bottom" />
        </div>
        <div className="silver-orbit silver-orbit-front" />
      </div>
      <div className="core-caption"><span /> Verification core</div>
    </div>
  );
}

function Home({ onSelect, onUpload }: { onSelect: (file: File) => void; onUpload: () => void }) {
  return (
    <>
      <main className="home-grid">
        <section className="hero-panel">
        <div className="hero-grid-lines" />
        <div className="hero-art" aria-hidden="true">
          <div className="tech-card tech-card-one"><Fingerprint size={62} /></div>
          <div className="tech-card tech-card-two"><ShieldCheck size={50} /></div>
          <div className="tech-card tech-card-three"><Eye size={48} /></div>
          <span className="circuit-line circuit-one" />
          <span className="circuit-line circuit-two" />
        </div>
        <div className="hero-scan-circle"><Fingerprint size={130} strokeWidth={0.45} /></div>
        <div className="relative z-10 max-w-2xl">
          <div className="section-label"><span /> Digital media verification</div>
          <h1 className="mt-8 text-[44px] font-semibold leading-[1.03] tracking-[-0.055em] text-white sm:text-6xl xl:text-[68px]">
            Detect What’s Fake.<br />
            <span className="text-gradient">Trust What You Share.</span>
          </h1>
          <p className="mt-7 max-w-xl text-sm leading-7 text-slate-400 sm:text-base">
            VeriSight AI inspects video for deepfake manipulation and scam signals—giving you a clear, evidence-based risk assessment.
          </p>
          <div className="mt-9 flex flex-wrap gap-3">
            <button className="primary-button" onClick={onUpload}>Analyze Video <ArrowRight size={15} /></button>
            <button className="outline-button px-5" onClick={onUpload}>How It Works <ArrowRight size={14} /></button>
          </div>
        </div>
        <VerificationCore />
        <div className="relative z-10 mt-16 flex flex-wrap items-center gap-6 border-t border-white/[0.06] pt-5 text-[9px] font-medium uppercase tracking-[0.15em] text-slate-600">
          <span className="flex items-center gap-2"><ScanLine size={13} className="text-blue-400" /> Multi-signal review</span>
          <span className="flex items-center gap-2"><Fingerprint size={13} className="text-violet-400" /> Deepfake detection</span>
          <span className="flex items-center gap-2"><Activity size={13} className="text-cyan-400" /> Scam analysis</span>
        </div>
        </section>

        <section className="verify-panel">
        <div className="mb-7 flex items-start justify-between gap-4">
          <div>
            <div className="section-label"><span /> VeriSight / Video verification</div>
            <h2 className="mt-5 text-2xl font-semibold tracking-tight text-white">Verify a Video</h2>
            <p className="mt-2 max-w-md text-xs leading-5 text-slate-500">Upload a video (MP4, MOV or WebM, English speech, up to 90 seconds). VeriSight analyzes the content and reports deepfake risk and scam risk separately.</p>
          </div>
          <button className="demo-tag" onClick={onUpload} title="Upload a video to run the live detector"><Sparkles size={11} /> Analyze a video</button>
        </div>
        <div className="signal-tabs">
          <button className="signal-tab signal-tab-active"><Upload size={13} /> Upload video</button>
        </div>
        <Dropzone onSelect={onSelect} compact />
        <div className="mt-5 flex items-center justify-between border-t border-white/[0.06] pt-4 text-[9px] uppercase tracking-[0.12em] text-slate-700">
          <span>No sign-up required</span><span>Max file 500 MB</span>
        </div>
        </section>
      </main>

      <section id="how-it-works" className="landing-section process-section">
        <div className="process-copy">
          <div className="section-label"><span /> A simple, traceable flow / 02</div>
          <h2>One video.<br /><span>Several<br />perspectives.</span></h2>
          <p>VeriSight follows a clear sequence, then checks different signals in parallel before bringing them together.</p>
          <button className="text-cta" onClick={onUpload}>Analyze a video <ArrowRight size={14} /></button>
        </div>
        <div className="process-map">
          <div className="process-row process-row-two">
            <ProcessNode number="01" title="Video input" text="Upload a short video clip" icon={FileVideo2} />
            <ArrowRight className="process-arrow" size={18} />
            <ProcessNode number="02" title="Preprocessing" text="Prepare frames and audio" icon={Sparkles} />
          </div>
          <div className="parallel-box">
            <span>03 / Analysis in parallel</span>
            <div>
              <div><Eye size={14} /> Visual + face analysis</div>
              <div><Mic2 size={14} /> Audio analysis</div>
              <div><Activity size={14} /> Speech + scam-language analysis</div>
            </div>
          </div>
          <div className="process-row process-row-two">
            <ProcessNode number="04" title="Combine the findings" text="Bring the signals together" icon={ScanLine} />
            <ArrowRight className="process-arrow" size={18} />
            <ProcessNode number="05" title="Final result" text="Deepfake risk and scam risk, separately" icon={ShieldCheck} />
          </div>
          <div className="process-note"><span /> Deepfake risk and scam risk are reported separately, with the reliability of each signal.</div>
        </div>
      </section>

      <section id="about" className="landing-section about-section">
        <div className="about-network" aria-hidden="true">
          {[Eye, Fingerprint, ShieldCheck, Activity, Mic2, ScanLine].map((Icon, index) => (
            <div key={index} className={`network-node network-node-${index + 1}`}><Icon size={24} /></div>
          ))}
        </div>
        <div className="about-content">
          <div className="about-icon"><ShieldCheck size={19} /></div>
          <div className="section-label justify-center"><span /> About VeriSight</div>
          <h2>A clearer way to<br /><span>verify what you watch.</span></h2>
          <p>VeriSight helps people assess whether videos may contain AI-generated or manipulated content, bringing multiple signals into one clear, understandable view.</p>
          <button className="primary-button" onClick={onUpload}>Analyze a video <ArrowRight size={15} /></button>
        </div>
        <footer className="landing-footer">
          <Brand />
          <span>AI-assisted media verification</span>
          <span>Results are predictions, not definitive proof.</span>
        </footer>
      </section>
    </>
  );
}

function ProcessNode({ number, title, text, icon: Icon }: { number: string; title: string; text: string; icon: typeof Eye }) {
  return (
    <div className="process-node">
      <div className="process-icon"><Icon size={16} /></div>
      <div><span>{number}</span><h3>{title}</h3><p>{text}</p></div>
    </div>
  );
}

function UploadPage({
  video,
  analyzing,
  stage,
  error,
  onSelect,
  onRemove,
  onAnalyze,
}: {
  video: SelectedVideo | null;
  analyzing: boolean;
  stage: string | null;
  error: string | null;
  onSelect: (file: File) => void;
  onRemove: () => void;
  onAnalyze: () => void;
}) {
  const steps = ["Preparing video", "Visual & voice", "Speech & language", "Risk assessment"];
  const info = STAGES.find((s) => stage && stage.includes(s.match));
  const analysisStep = info?.step ?? 0;
  const stageLabel = info?.label ?? (analyzing ? "Working…" : null);

  return (
    <main className="page-wrap">
      <div className="mx-auto w-full max-w-4xl">
        <div className="mb-8 text-center">
          <div className="section-label justify-center"><span /> Secure verification console</div>
          <h1 className="mt-4 text-3xl font-semibold tracking-tight text-white sm:text-4xl">Analyze a video</h1>
          <p className="mt-3 text-sm text-slate-500">Checks the video for deepfake and scam-related signals. English videos, up to 90 seconds.</p>
        </div>

        {!video ? (
          <Dropzone onSelect={onSelect} />
        ) : (
          <div className="selected-grid">
            <div className="video-frame">
              <video src={video.url} className="size-full object-contain" controls />
              <div className="absolute left-3 top-3 flex items-center gap-2 rounded bg-black/70 px-2.5 py-1.5 text-[8px] uppercase tracking-[0.14em] text-emerald-300 backdrop-blur">
                <span className="status-dot" /> Ready
              </div>
              <span className="corner corner-tl" /><span className="corner corner-tr" /><span className="corner corner-bl" /><span className="corner corner-br" />
            </div>
            <div className="flex flex-col p-6">
              <div className="flex size-10 items-center justify-center rounded-lg border border-blue-400/15 bg-blue-400/[0.07] text-blue-300"><Video size={18} /></div>
              <div className="mt-5 break-words text-sm font-medium text-white">{video.name}</div>
              <div className="mt-2 text-[10px] uppercase tracking-[0.12em] text-slate-600">{video.size} · Secure upload</div>
              <div className="mt-7 space-y-3">
                {["Visual appearance", "Voice analysis", "Scam language"].map((item) => (
                  <div key={item} className="flex items-center gap-2 border-b border-white/[0.05] pb-3 text-[11px] text-slate-500">
                    <Check size={12} className="text-blue-400" /> {item}
                  </div>
                ))}
              </div>
              <button className="text-button mt-4" disabled={analyzing} onClick={onRemove}><X size={13} /> Remove video</button>
            </div>
          </div>
        )}

        <button disabled={!video || analyzing} className="primary-button mt-4 w-full justify-center py-3.5 disabled:cursor-not-allowed disabled:opacity-40" onClick={onAnalyze}>
          {analyzing ? <><ScanLine size={16} className="animate-pulse" /> Analyzing…</> : <><Play size={15} fill="currentColor" /> Analyze Video</>}
        </button>
        {analyzing && (
          <>
            <div className="analysis-track"><span /></div>
            <div className="analysis-steps">
              {steps.map((step, index) => (
                <div key={step} className={index <= analysisStep ? "analysis-step-active" : ""}>
                  <span>{index < analysisStep ? <Check size={10} /> : index + 1}</span>{step}
                </div>
              ))}
            </div>
            {stageLabel && <p className="mt-3 text-center text-[11px] text-slate-500">{stageLabel}</p>}
          </>
        )}
        {error && (
          <div className="notice mt-4 justify-center">
            <TriangleAlert size={13} /> {error}
          </div>
        )}
      </div>
    </main>
  );
}

export default function App() {
  const [page, setPage] = useState<Page>("home");
  const [video, setVideo] = useState<SelectedVideo | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [stage, setStage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!analyzing) return;
    const tick = window.setInterval(() => {
      fetchProgress()
        .then((p) => { if (p.stage) setStage(p.stage); })
        .catch(() => undefined);
    }, 700);
    return () => window.clearInterval(tick);
  }, [analyzing]);

  useEffect(() => () => {
    if (video?.url) URL.revokeObjectURL(video.url);
  }, [video]);

  const selectVideo = (file: File) => {
    const ext = file.name.includes(".") ? `.${file.name.split(".").pop()!.toLowerCase()}` : "";
    if (!ACCEPTED_EXT.includes(ext)) {
      setError("Please upload an MP4, MOV or WebM video.");
      return;
    }
    if (video?.url) URL.revokeObjectURL(video.url);
    setResult(null);
    setError(null);
    setVideo({
      name: file.name,
      size: file.size > 1024 * 1024 ? `${(file.size / 1024 / 1024).toFixed(1)} MB` : `${Math.round(file.size / 1024)} KB`,
      url: URL.createObjectURL(file),
      file,
    });
    setPage("upload");
  };

  const analyze = () => {
    if (!video) return;
    setResult(null);
    setAnalyzing(true);
    setStage("Uploading");
    setError(null);
    analyzeVideo(video.file)
      .then((r) => {
        console.log("REAL API RESPONSE:", r);
        setResult(r);
        setPage("results");
      })
      .catch((e) => {
        setResult(null);
        setError(e instanceof ApiError ? e.message : "Analysis unavailable. Something went wrong while analysing this video.");
        setPage("upload");
      })
      .finally(() => {
        setAnalyzing(false);
        setStage(null);
      });
  };

  const reset = () => {
    if (video?.url) URL.revokeObjectURL(video.url);
    setVideo(null);
    setResult(null);
    setError(null);
    setAnalyzing(false);
    setStage(null);
    setPage("upload");
  };

  return (
    <div className="app-shell">
      <Header page={page} onNavigate={(p) => { if (p !== "results" || result) setPage(p); }} />
      {page === "home" && <Home onSelect={selectVideo} onUpload={() => setPage("upload")} />}
      {page === "upload" && (
        <UploadPage
          video={video}
          analyzing={analyzing}
          stage={stage}
          error={error}
          onSelect={selectVideo}
          onRemove={() => { if (video?.url) URL.revokeObjectURL(video.url); setVideo(null); setError(null); }}
          onAnalyze={analyze}
        />
      )}
      {page === "results" && result && (
        <ResultsPage result={result} video={video ? { name: video.name, url: video.url } : null} onAgain={reset} />
      )}
    </div>
  );
}
