import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Download, FlipHorizontal2, Play, Trophy, X } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { useCallback, useEffect, useRef, useState } from "react";

// ─── TensorFlow.js + MoveNet loaded via CDN (index.html script tags) ───
declare const tf: {
  setBackend: (backend: string) => Promise<void>;
  ready: () => Promise<void>;
};

declare const poseDetection: {
  createDetector: (
    model: string,
    config: Record<string, unknown>,
  ) => Promise<PoseDetector>;
  SupportedModels: { MoveNet: string };
  movenet: {
    modelType: { SINGLEPOSE_THUNDER: string; SINGLEPOSE_LIGHTNING: string };
  };
};

interface Keypoint {
  x: number;
  y: number;
  score: number;
  name?: string;
}

interface Pose {
  keypoints: Keypoint[];
  score?: number;
}

interface PoseDetector {
  estimatePoses: (input: HTMLVideoElement) => Promise<Pose[]>;
  dispose: () => void;
}

interface LockedVerdict {
  verdict: "legal" | "noball";
  angle: number;
  timestamp: string;
  confidence: number;
}

interface TournamentState {
  teamA: string;
  teamB: string;
  bowler: string;
}

interface BowlerStats {
  total: number;
  legal: number;
  noBall: number;
}

// ─── Skeleton connections (MoveNet 17-keypoint) ───
const SKELETON_PAIRS: [number, number][] = [
  [0, 1],
  [0, 2],
  [1, 3],
  [2, 4],
  [5, 6],
  [5, 7],
  [7, 9],
  [6, 8],
  [8, 10],
  [5, 11],
  [6, 12],
  [11, 12],
  [11, 13],
  [13, 15],
  [12, 14],
  [14, 16],
];

const LEG_INDICES = new Set([11, 12, 13, 14, 15, 16]);
const CONFIDENCE_THRESHOLD = 0.3;
const FULL_BODY_KEYPOINTS = [5, 6, 11, 12, 13, 14, 15, 16];
const FULL_BODY_SCORE = 0.35;
const ANGLE_BUFFER_SIZE = 8;
const STABILITY_DOTS = [
  "d0",
  "d1",
  "d2",
  "d3",
  "d4",
  "d5",
  "d6",
  "d7",
] as const;
const STABILITY_MIN_READINGS = 5;
const STABILITY_MAX_STD = 8;
const FRAME_BUFFER_MAX = 60;
const REPLAY_FRAMES = 40;

function calculateAngle(a: Keypoint, b: Keypoint, c: Keypoint): number {
  const v1 = { x: a.x - b.x, y: a.y - b.y };
  const v2 = { x: c.x - b.x, y: c.y - b.y };
  const dot = v1.x * v2.x + v1.y * v2.y;
  const mag1 = Math.sqrt(v1.x * v1.x + v1.y * v1.y);
  const mag2 = Math.sqrt(v2.x * v2.x + v2.y * v2.y);
  if (mag1 === 0 || mag2 === 0) return 180;
  const cosA = Math.max(-1, Math.min(1, dot / (mag1 * mag2)));
  return (Math.acos(cosA) * 180) / Math.PI;
}

function isFullBodyVisible(keypoints: Keypoint[]): boolean {
  return FULL_BODY_KEYPOINTS.every(
    (idx) => (keypoints[idx]?.score || 0) >= FULL_BODY_SCORE,
  );
}

function stdDev(arr: number[]): number {
  if (arr.length === 0) return 0;
  const mean = arr.reduce((s, v) => s + v, 0) / arr.length;
  const variance = arr.reduce((s, v) => s + (v - mean) ** 2, 0) / arr.length;
  return Math.sqrt(variance);
}

function computeConfidence(
  hip: Keypoint,
  knee: Keypoint,
  ankle: Keypoint,
  bufLen: number,
): number {
  const avgScore =
    ((hip.score || 0) + (knee.score || 0) + (ankle.score || 0)) / 3;
  const stabilityRatio = Math.min(bufLen / 8, 1);
  return Math.round(avgScore * stabilityRatio * 100);
}

function playNoBallSound(audioCtx: AudioContext) {
  const osc1 = audioCtx.createOscillator();
  const gain = audioCtx.createGain();
  osc1.connect(gain);
  gain.connect(audioCtx.destination);
  osc1.frequency.setValueAtTime(440, audioCtx.currentTime);
  osc1.frequency.linearRampToValueAtTime(880, audioCtx.currentTime + 0.4);
  osc1.frequency.linearRampToValueAtTime(440, audioCtx.currentTime + 0.8);
  gain.gain.setValueAtTime(0.3, audioCtx.currentTime);
  gain.gain.linearRampToValueAtTime(0, audioCtx.currentTime + 0.9);
  osc1.start();
  osc1.stop(audioCtx.currentTime + 0.9);
}

function playLegalSound(audioCtx: AudioContext) {
  const osc = audioCtx.createOscillator();
  const gain = audioCtx.createGain();
  osc.connect(gain);
  gain.connect(audioCtx.destination);
  osc.frequency.value = 880;
  gain.gain.setValueAtTime(0.2, audioCtx.currentTime);
  gain.gain.linearRampToValueAtTime(0, audioCtx.currentTime + 0.15);
  osc.start();
  osc.stop(audioCtx.currentTime + 0.15);
}

// ─── Replay Modal ───
function ReplayModal({
  frames,
  onClose,
}: {
  frames: string[];
  onClose: () => void;
}) {
  const [frameIdx, setFrameIdx] = useState(0);
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;

  useEffect(() => {
    if (frames.length === 0) {
      onCloseRef.current();
      return;
    }
    let idx = 0;
    let loops = 0;
    let closed = false;

    const interval = setInterval(() => {
      if (closed) return;
      idx += 1;
      if (idx >= frames.length) {
        loops += 1;
        if (loops >= 2) {
          clearInterval(interval);
          closed = true;
          onCloseRef.current();
          return;
        }
        idx = 0;
      }
      setFrameIdx(idx);
    }, 120);

    return () => {
      closed = true;
      clearInterval(interval);
    };
  }, [frames]);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/90 backdrop-blur-sm"
    >
      <div className="relative max-w-2xl w-full mx-4">
        <div className="text-center mb-3">
          <span className="text-primary font-display font-black text-xl uppercase tracking-widest animate-glow">
            🎬 SLOW MOTION REPLAY
          </span>
        </div>
        <div className="relative rounded-2xl overflow-hidden border-2 border-primary/40">
          {frames[frameIdx] ? (
            <img
              src={frames[frameIdx]}
              className="w-full block"
              alt={`replay frame ${frameIdx}`}
            />
          ) : (
            <div className="aspect-video bg-card flex items-center justify-center">
              <span className="text-muted-foreground">No frames</span>
            </div>
          )}
          <div className="absolute bottom-3 left-1/2 -translate-x-1/2 bg-black/70 rounded-full px-3 py-1 text-xs font-mono text-primary">
            {frameIdx + 1} / {frames.length}
          </div>
        </div>
        <button
          type="button"
          data-ocid="replay.close_button"
          onClick={() => onCloseRef.current()}
          className="absolute top-8 right-2 p-2 bg-black/60 rounded-full border border-white/20 text-white hover:bg-black/80 transition-colors"
        >
          <X size={16} />
        </button>
      </div>
    </motion.div>
  );
}

// ─── Clip Modal ───
function ClipModal({
  clipUrl,
  onClose,
}: {
  clipUrl: string;
  onClose: () => void;
}) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/90 backdrop-blur-sm"
    >
      <div className="relative max-w-2xl w-full mx-4">
        <div className="text-center mb-3">
          <span className="text-primary font-display font-black text-lg uppercase tracking-widest">
            📹 Delivery Clip
          </span>
        </div>
        <video
          src={clipUrl}
          controls
          autoPlay
          loop
          className="w-full rounded-2xl border border-primary/30"
        >
          <track kind="captions" />
        </video>
        <div className="flex gap-3 mt-3">
          <a
            href={clipUrl}
            download="delivery-clip.webm"
            data-ocid="clip.download_button"
            className="flex-1 flex items-center justify-center gap-2 py-2.5 bg-primary text-primary-foreground rounded-xl font-bold text-sm hover:opacity-90 transition-opacity"
          >
            <Download size={16} /> Download
          </a>
          <button
            type="button"
            data-ocid="clip.close_button"
            onClick={onClose}
            className="flex-1 py-2.5 bg-secondary text-secondary-foreground rounded-xl font-bold text-sm hover:opacity-90 transition-opacity border border-border"
          >
            Close
          </button>
        </div>
      </div>
    </motion.div>
  );
}

type AppStatus = "idle" | "loading" | "ready" | "running" | "stopped" | "error";
type Verdict = "legal" | "noball" | "detecting" | null;

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const animFrameRef = useRef<number | null>(null);
  const detectorRef = useRef<PoseDetector | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const isRunningRef = useRef(false);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordingChunksRef = useRef<Blob[]>([]);
  const lastClipUrlRef = useRef<string | null>(null);
  const isDraggingCreaseRef = useRef(false);
  const creaseLineYRef = useRef(65);
  const activeTournamentRef = useRef<TournamentState | null>(null);

  // Stability & delivery detection
  const angleBufferRef = useRef<number[]>([]);
  const minAngleInWindowRef = useRef<number>(180);
  const prevAngleRef = useRef<number | null>(null);
  const frameBufferRef = useRef<string[]>([]);

  const [status, setStatus] = useState<AppStatus>("idle");
  const [verdict, setVerdict] = useState<Verdict>(null);
  const [kneeAngle, setKneeAngle] = useState<number | null>(null);
  const [errorMsg, setErrorMsg] = useState<string>("");
  const [tfReady, setTfReady] = useState(false);
  const [facingMode, setFacingMode] = useState<"environment" | "user">(
    "environment",
  );
  const [stabilityCount, setStabilityCount] = useState(0);
  const [lockedVerdict, setLockedVerdict] = useState<LockedVerdict | null>(
    null,
  );

  // Feature states
  const [ballReleasedFlash, setBallReleasedFlash] = useState(false);
  const [replayFrames, setReplayFrames] = useState<string[]>([]);
  const [showReplay, setShowReplay] = useState(false);
  const [lastClipUrl, setLastClipUrl] = useState<string | null>(null);
  const [showClipModal, setShowClipModal] = useState(false);
  const [creaseLineY, setCreaseLineY] = useState(65);
  const [footOverCrease, setFootOverCrease] = useState(false);
  const [modelCached, setModelCached] = useState(() => {
    try {
      return localStorage.getItem("movenet_cached") === "1";
    } catch {
      return false;
    }
  });
  const [totalBalls, setTotalBalls] = useState(0);
  const [legalBalls, setLegalBalls] = useState(0);
  const [noBalls, setNoBalls] = useState(0);
  const [tournamentOpen, setTournamentOpen] = useState(false);
  const [teamAInput, setTeamAInput] = useState("");
  const [teamBInput, setTeamBInput] = useState("");
  const [bowlerInput, setBowlerInput] = useState("");
  const [activeTournament, setActiveTournament] =
    useState<TournamentState | null>(null);
  const [bowlerStats, setBowlerStats] = useState<Record<string, BowlerStats>>(
    {},
  );

  // Create capture canvas once
  useEffect(() => {
    captureCanvasRef.current = document.createElement("canvas");
  }, []);

  // Keep activeTournamentRef in sync
  useEffect(() => {
    activeTournamentRef.current = activeTournament;
  }, [activeTournament]);

  // ── Lazy AudioContext init ──
  const ensureAudioCtx = useCallback(() => {
    if (!audioCtxRef.current) {
      audioCtxRef.current = new AudioContext();
    }
    return audioCtxRef.current;
  }, []);

  // ── MediaRecorder helpers ──
  const startRecording = useCallback((stream: MediaStream) => {
    if (typeof MediaRecorder === "undefined") return;
    const mimeType = MediaRecorder.isTypeSupported("video/webm;codecs=vp9")
      ? "video/webm;codecs=vp9"
      : MediaRecorder.isTypeSupported("video/webm")
        ? "video/webm"
        : MediaRecorder.isTypeSupported("video/mp4")
          ? "video/mp4"
          : null;
    if (!mimeType) return;

    try {
      recordingChunksRef.current = [];
      const mr = new MediaRecorder(stream, { mimeType });

      mr.ondataavailable = (e) => {
        if (e.data.size > 0) recordingChunksRef.current.push(e.data);
      };

      mr.onstop = () => {
        const blob = new Blob(recordingChunksRef.current, { type: mimeType });
        const url = URL.createObjectURL(blob);
        if (lastClipUrlRef.current) URL.revokeObjectURL(lastClipUrlRef.current);
        lastClipUrlRef.current = url;
        setLastClipUrl(url);
        recordingChunksRef.current = [];
        // Restart if still running
        if (isRunningRef.current && streamRef.current) {
          startRecording(streamRef.current);
        }
      };

      mr.start();
      mediaRecorderRef.current = mr;
    } catch (e) {
      console.warn("MediaRecorder error:", e);
    }
  }, []);

  const stopRecordingAndSave = useCallback(() => {
    const mr = mediaRecorderRef.current;
    if (mr && mr.state === "recording") {
      mr.stop();
    }
  }, []);

  // ── Load TF + MoveNet model on mount ──
  useEffect(() => {
    let cancelled = false;

    async function loadModel() {
      setStatus("loading");
      try {
        let retries = 0;
        while (
          (typeof tf === "undefined" || typeof poseDetection === "undefined") &&
          retries < 20
        ) {
          await new Promise((r) => setTimeout(r, 500));
          retries++;
        }
        if (typeof tf === "undefined" || typeof poseDetection === "undefined") {
          throw new Error("TensorFlow.js CDN scripts not loaded");
        }

        await tf.setBackend("webgl");
        await tf.ready();

        const detector = await poseDetection.createDetector(
          poseDetection.SupportedModels.MoveNet,
          {
            modelType: poseDetection.movenet.modelType.SINGLEPOSE_THUNDER,
            enableSmoothing: true,
          },
        );

        if (!cancelled) {
          detectorRef.current = detector;
          setTfReady(true);
          setStatus("ready");
          // Mark model as cached for offline use
          try {
            localStorage.setItem("movenet_cached", "1");
            setModelCached(true);
          } catch {
            // ignore
          }
        }
      } catch (err) {
        if (!cancelled) {
          console.error("Model load error:", err);
          setStatus("error");
          setErrorMsg("Failed to load AI model. Please refresh and try again.");
        }
      }
    }

    loadModel();
    return () => {
      cancelled = true;
    };
  }, []);

  // ── Draw pose skeleton on canvas ──
  const drawPose = useCallback(
    (keypoints: Keypoint[], videoWidth: number, videoHeight: number) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      canvas.width = videoWidth;
      canvas.height = videoHeight;
      ctx.clearRect(0, 0, videoWidth, videoHeight);

      // Draw crease line on canvas
      const creaseY = creaseLineYRef.current;
      const cyPx = (creaseY / 100) * videoHeight;
      ctx.save();
      ctx.strokeStyle = "rgba(255, 60, 60, 0.85)";
      ctx.lineWidth = 2;
      ctx.setLineDash([12, 6]);
      ctx.shadowColor = "#ff3c3c";
      ctx.shadowBlur = 8;
      ctx.beginPath();
      ctx.moveTo(0, cyPx);
      ctx.lineTo(videoWidth, cyPx);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.restore();

      ctx.lineWidth = 3;
      for (const [i, j] of SKELETON_PAIRS) {
        const kp1 = keypoints[i];
        const kp2 = keypoints[j];
        if (!kp1 || !kp2) continue;
        if (
          (kp1.score || 0) < CONFIDENCE_THRESHOLD ||
          (kp2.score || 0) < CONFIDENCE_THRESHOLD
        )
          continue;

        const isLegLine = LEG_INDICES.has(i) && LEG_INDICES.has(j);
        ctx.strokeStyle = isLegLine ? "#fbbf24" : "#22d3ee";
        ctx.shadowColor = isLegLine ? "#fbbf24" : "#22d3ee";
        ctx.shadowBlur = 10;
        ctx.beginPath();
        ctx.moveTo(kp1.x, kp1.y);
        ctx.lineTo(kp2.x, kp2.y);
        ctx.stroke();
      }

      ctx.shadowBlur = 0;

      for (let i = 0; i < keypoints.length; i++) {
        const kp = keypoints[i];
        if (!kp || (kp.score || 0) < CONFIDENCE_THRESHOLD) continue;

        const isLeg = LEG_INDICES.has(i);
        const isKnee = i === 13 || i === 14;

        ctx.shadowColor = isLeg ? "#fbbf24" : "#22d3ee";
        ctx.shadowBlur = isKnee ? 20 : 12;
        ctx.fillStyle = isKnee ? "#f97316" : isLeg ? "#fbbf24" : "#22d3ee";
        ctx.beginPath();
        ctx.arc(kp.x, kp.y, isKnee ? 10 : isLeg ? 8 : 5, 0, Math.PI * 2);
        ctx.fill();

        if (isKnee) {
          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth = 2;
          ctx.shadowBlur = 0;
          ctx.beginPath();
          ctx.arc(kp.x, kp.y, 14, 0, Math.PI * 2);
          ctx.stroke();
        }
      }

      ctx.shadowBlur = 0;
    },
    [],
  );

  // ── Capture composite frame for replay ──
  const captureFrame = useCallback(
    (videoEl: HTMLVideoElement, vw: number, vh: number) => {
      const cc = captureCanvasRef.current;
      const overlayCanvas = canvasRef.current;
      if (!cc || !overlayCanvas) return;
      cc.width = vw;
      cc.height = vh;
      const cctx = cc.getContext("2d");
      if (!cctx) return;
      cctx.drawImage(videoEl, 0, 0, vw, vh);
      cctx.drawImage(overlayCanvas, 0, 0);
      const dataUrl = cc.toDataURL("image/jpeg", 0.5);
      frameBufferRef.current.push(dataUrl);
      if (frameBufferRef.current.length > FRAME_BUFFER_MAX) {
        frameBufferRef.current.shift();
      }
    },
    [],
  );

  // ── Main detection loop ──
  const runDetection = useCallback(() => {
    if (!isRunningRef.current) return;

    const video = videoRef.current;
    const detector = detectorRef.current;

    if (!video || !detector || video.readyState < 2) {
      animFrameRef.current = requestAnimationFrame(runDetection);
      return;
    }

    detector
      .estimatePoses(video)
      .then((poses) => {
        if (!isRunningRef.current) return;

        const vw = video.videoWidth || 640;
        const vh = video.videoHeight || 480;

        if (poses.length > 0) {
          const kps = poses[0].keypoints;
          drawPose(kps, vw, vh);
          captureFrame(video, vw, vh);

          // Full body detection gate
          if (!isFullBodyVisible(kps)) {
            setVerdict("detecting");
            setKneeAngle(null);
            setFootOverCrease(false);
            angleBufferRef.current = [];
            setStabilityCount(0);
            animFrameRef.current = requestAnimationFrame(runDetection);
            return;
          }

          // Pick the leg side with higher combined confidence
          const lHip = kps[11];
          const lKnee = kps[13];
          const lAnkle = kps[15];
          const rHip = kps[12];
          const rKnee = kps[14];
          const rAnkle = kps[16];

          const lConf =
            (lHip?.score || 0) + (lKnee?.score || 0) + (lAnkle?.score || 0);
          const rConf =
            (rHip?.score || 0) + (rKnee?.score || 0) + (rAnkle?.score || 0);

          const useSide = lConf >= rConf ? "left" : "right";
          const hip = useSide === "left" ? lHip : rHip;
          const knee = useSide === "left" ? lKnee : rKnee;
          const ankle = useSide === "left" ? lAnkle : rAnkle;

          // Crease line check
          if (ankle && (ankle.score || 0) > CONFIDENCE_THRESHOLD) {
            const ankleYPercent = (ankle.y / vh) * 100;
            setFootOverCrease(ankleYPercent > creaseLineYRef.current);
          }

          if (
            hip?.score > CONFIDENCE_THRESHOLD &&
            knee?.score > CONFIDENCE_THRESHOLD &&
            ankle?.score > CONFIDENCE_THRESHOLD
          ) {
            const angle = calculateAngle(hip, knee, ankle);
            const rounded = Math.round(angle);
            setKneeAngle(rounded);

            // Push to stability buffer
            const buf = angleBufferRef.current;
            buf.push(rounded);
            if (buf.length > ANGLE_BUFFER_SIZE) buf.shift();
            setStabilityCount(buf.length);

            const sd = stdDev(buf);
            const isStable =
              buf.length >= STABILITY_MIN_READINGS && sd < STABILITY_MAX_STD;

            if (isStable) {
              const liveVerdict: "legal" | "noball" =
                rounded < 160 ? "noball" : "legal";
              setVerdict(liveVerdict);

              // Delivery phase lock: track minimum angle
              if (rounded <= (minAngleInWindowRef.current ?? rounded)) {
                minAngleInWindowRef.current = rounded;
              } else if (
                prevAngleRef.current !== null &&
                rounded > minAngleInWindowRef.current + 5
              ) {
                // Angle increasing > 5° from min → delivery moment, lock verdict
                const lockedAngle = minAngleInWindowRef.current;
                const now = new Date();
                const ts = now.toLocaleTimeString("en-GB", {
                  hour: "2-digit",
                  minute: "2-digit",
                  second: "2-digit",
                });
                const v: "legal" | "noball" =
                  lockedAngle < 160 ? "noball" : "legal";
                const conf = computeConfidence(hip, knee, ankle, buf.length);

                setLockedVerdict({
                  verdict: v,
                  angle: lockedAngle,
                  timestamp: ts,
                  confidence: conf,
                });

                // Ball release flash
                setBallReleasedFlash(true);
                setTimeout(() => setBallReleasedFlash(false), 1500);

                // Sound effects
                const actx = audioCtxRef.current;
                if (actx) {
                  if (v === "noball") {
                    playNoBallSound(actx);
                  } else {
                    playLegalSound(actx);
                  }
                }

                // Delivery counters
                setTotalBalls((p) => p + 1);
                if (v === "legal") {
                  setLegalBalls((p) => p + 1);
                } else {
                  setNoBalls((p) => p + 1);
                }

                // Tournament stats
                const tournament = activeTournamentRef.current;
                if (tournament) {
                  const bowler = tournament.bowler;
                  setBowlerStats((prev) => ({
                    ...prev,
                    [bowler]: {
                      total: (prev[bowler]?.total || 0) + 1,
                      legal:
                        (prev[bowler]?.legal || 0) + (v === "legal" ? 1 : 0),
                      noBall:
                        (prev[bowler]?.noBall || 0) + (v === "noball" ? 1 : 0),
                    },
                  }));
                }

                // Capture replay frames
                const capturedFrames = frameBufferRef.current.slice(
                  -REPLAY_FRAMES,
                );
                setReplayFrames(capturedFrames);
                if (capturedFrames.length > 0) {
                  setShowReplay(true);
                }

                // Stop recording to save clip
                stopRecordingAndSave();

                // Reset window
                minAngleInWindowRef.current = 180;
                angleBufferRef.current = [];
                setStabilityCount(0);
              }

              prevAngleRef.current = rounded;
            } else {
              setVerdict("detecting");
            }
          } else {
            setVerdict("detecting");
            setKneeAngle(null);
          }
        } else {
          setVerdict("detecting");
          setKneeAngle(null);
          setFootOverCrease(false);
          angleBufferRef.current = [];
          setStabilityCount(0);
        }

        animFrameRef.current = requestAnimationFrame(runDetection);
      })
      .catch(() => {
        if (isRunningRef.current) {
          animFrameRef.current = requestAnimationFrame(runDetection);
        }
      });
  }, [drawPose, captureFrame, stopRecordingAndSave]);

  // ── Stop camera ──
  const stopCamera = useCallback(() => {
    isRunningRef.current = false;
    if (animFrameRef.current !== null) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = null;
    }
    if (
      mediaRecorderRef.current &&
      mediaRecorderRef.current.state !== "inactive"
    ) {
      mediaRecorderRef.current.stop();
    }
    if (streamRef.current) {
      for (const t of streamRef.current.getTracks()) t.stop();
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      ctx?.clearRect(0, 0, canvas.width, canvas.height);
    }
    angleBufferRef.current = [];
    frameBufferRef.current = [];
    minAngleInWindowRef.current = 180;
    prevAngleRef.current = null;
    setStatus("stopped");
    setVerdict(null);
    setKneeAngle(null);
    setStabilityCount(0);
    setLockedVerdict(null);
    setFootOverCrease(false);
  }, []);

  // ── Start camera with given facing mode ──
  const startCamera = useCallback(
    async (mode?: "environment" | "user") => {
      if (!tfReady || isRunningRef.current) return;
      const facing = mode ?? facingMode;
      setErrorMsg("");
      // Init AudioContext on user interaction
      ensureAudioCtx();
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: facing,
            width: { ideal: 640 },
            height: { ideal: 480 },
          },
        });
        streamRef.current = stream;
        const video = videoRef.current;
        if (!video) return;
        video.srcObject = stream;
        await video.play();
        isRunningRef.current = true;
        setStatus("running");
        setVerdict("detecting");
        setLockedVerdict(null);
        frameBufferRef.current = [];
        // Start recording
        startRecording(stream);
        runDetection();
      } catch (err: unknown) {
        const e = err as { name?: string };
        setStatus("error");
        if (
          e?.name === "NotAllowedError" ||
          e?.name === "PermissionDeniedError"
        ) {
          setErrorMsg(
            "Camera permission denied. Please allow camera access and try again.",
          );
        } else {
          setErrorMsg(
            "Could not access camera. Please check your device and browser settings.",
          );
        }
      }
    },
    [tfReady, facingMode, runDetection, ensureAudioCtx, startRecording],
  );

  // ── Swap camera ──
  const swapCamera = useCallback(async () => {
    const newMode = facingMode === "environment" ? "user" : "environment";
    setFacingMode(newMode);
    if (isRunningRef.current) {
      isRunningRef.current = false;
      if (animFrameRef.current !== null) {
        cancelAnimationFrame(animFrameRef.current);
        animFrameRef.current = null;
      }
      if (
        mediaRecorderRef.current &&
        mediaRecorderRef.current.state !== "inactive"
      ) {
        mediaRecorderRef.current.stop();
      }
      if (streamRef.current) {
        for (const t of streamRef.current.getTracks()) t.stop();
        streamRef.current = null;
      }
      if (videoRef.current) videoRef.current.srcObject = null;
      angleBufferRef.current = [];
      frameBufferRef.current = [];
      minAngleInWindowRef.current = 180;
      prevAngleRef.current = null;
      setStabilityCount(0);
      await startCamera(newMode);
    }
  }, [facingMode, startCamera]);

  const retryLoad = useCallback(() => {
    window.location.reload();
  }, []);

  // ── Cleanup on unmount ──
  useEffect(() => {
    return () => {
      stopCamera();
      detectorRef.current?.dispose();
      if (lastClipUrlRef.current) URL.revokeObjectURL(lastClipUrlRef.current);
    };
  }, [stopCamera]);

  // ── Crease line drag handlers ──
  const handleCreasePointerDown = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      e.stopPropagation();
      isDraggingCreaseRef.current = true;
      (e.target as Element).setPointerCapture(e.pointerId);
    },
    [],
  );

  const handleStagePointerMove = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      if (!isDraggingCreaseRef.current) return;
      const rect = stageRef.current?.getBoundingClientRect();
      if (!rect) return;
      const y = ((e.clientY - rect.top) / rect.height) * 100;
      const clamped = Math.max(20, Math.min(90, y));
      creaseLineYRef.current = clamped;
      setCreaseLineY(clamped);
    },
    [],
  );

  const handleStagePointerUp = useCallback(() => {
    isDraggingCreaseRef.current = false;
  }, []);

  // ── Tournament handlers ──
  const handleStartMatch = useCallback(() => {
    if (!teamAInput.trim() || !teamBInput.trim() || !bowlerInput.trim()) return;
    const t: TournamentState = {
      teamA: teamAInput.trim(),
      teamB: teamBInput.trim(),
      bowler: bowlerInput.trim(),
    };
    setActiveTournament(t);
    activeTournamentRef.current = t;
    setTournamentOpen(false);
  }, [teamAInput, teamBInput, bowlerInput]);

  const isRunning = status === "running";
  const isLoading = status === "loading";
  const isMirrored = facingMode === "user";

  const isStable =
    stabilityCount >= STABILITY_MIN_READINGS &&
    stdDev(angleBufferRef.current) < STABILITY_MAX_STD;

  const statusMessage = (() => {
    if (status === "loading")
      return "⚙️ Loading TensorFlow.js MoveNet THUNDER — please wait...";
    if (status === "error") return null;
    if (status === "ready")
      return "✅ AI Model ready · MoveNet THUNDER loaded · Click Start Camera";
    if (status === "stopped") return "⏹ Camera stopped · Ready to start again";
    if (status === "idle") return "Initialising...";
    if (lockedVerdict) return `Decision locked · ${lockedVerdict.timestamp}`;
    if (verdict === "detecting")
      return "Stand in frame — position your full body for detection";
    if (!isStable)
      return `Stabilising pose... ${stabilityCount}/${STABILITY_MIN_READINGS} readings`;
    return "Pose stable — tracking knee angle";
  })();

  const currentBowlerStats = activeTournament
    ? bowlerStats[activeTournament.bowler]
    : null;

  return (
    <div className="min-h-dvh bg-background text-foreground flex flex-col font-body">
      {/* ── Replay Overlay ── */}
      <AnimatePresence>
        {showReplay && replayFrames.length > 0 && (
          <ReplayModal
            frames={replayFrames}
            onClose={() => setShowReplay(false)}
          />
        )}
      </AnimatePresence>

      {/* ── Clip Modal ── */}
      <AnimatePresence>
        {showClipModal && lastClipUrl && (
          <ClipModal
            clipUrl={lastClipUrl}
            onClose={() => setShowClipModal(false)}
          />
        )}
      </AnimatePresence>

      {/* ── Tournament Dialog ── */}
      <Dialog open={tournamentOpen} onOpenChange={setTournamentOpen}>
        <DialogContent
          data-ocid="tournament.dialog"
          className="bg-card border-border"
        >
          <DialogHeader>
            <DialogTitle className="font-display text-xl flex items-center gap-2">
              <Trophy size={20} className="text-primary" /> Tournament Mode
            </DialogTitle>
          </DialogHeader>
          <div className="grid gap-4 py-2">
            <div className="grid gap-1.5">
              <Label htmlFor="teamA" className="text-sm font-bold">
                Team A Name
              </Label>
              <Input
                id="teamA"
                data-ocid="tournament.input"
                value={teamAInput}
                onChange={(e) => setTeamAInput(e.target.value)}
                placeholder="e.g. Street Warriors"
                className="bg-background border-border"
              />
            </div>
            <div className="grid gap-1.5">
              <Label htmlFor="teamB" className="text-sm font-bold">
                Team B Name
              </Label>
              <Input
                id="teamB"
                value={teamBInput}
                onChange={(e) => setTeamBInput(e.target.value)}
                placeholder="e.g. Garden Gladiators"
                className="bg-background border-border"
              />
            </div>
            <div className="grid gap-1.5">
              <Label htmlFor="bowler" className="text-sm font-bold">
                Current Bowler
              </Label>
              <Input
                id="bowler"
                value={bowlerInput}
                onChange={(e) => setBowlerInput(e.target.value)}
                placeholder="e.g. Ravi Kumar"
                className="bg-background border-border"
              />
            </div>
          </div>
          <DialogFooter className="gap-2">
            <Button
              variant="outline"
              data-ocid="tournament.cancel_button"
              onClick={() => setTournamentOpen(false)}
            >
              Cancel
            </Button>
            <Button
              data-ocid="tournament.submit_button"
              onClick={handleStartMatch}
              disabled={
                !teamAInput.trim() || !teamBInput.trim() || !bowlerInput.trim()
              }
              className="bg-primary text-primary-foreground hover:opacity-90"
            >
              🏆 Start Match
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* ── Header ── */}
      <header className="border-b border-border py-4 px-6 relative overflow-hidden">
        <div
          className="absolute inset-0 opacity-5"
          style={{
            backgroundImage:
              "repeating-linear-gradient(0deg, transparent, transparent 40px, oklch(0.5 0.1 155) 40px, oklch(0.5 0.1 155) 41px), repeating-linear-gradient(90deg, transparent, transparent 40px, oklch(0.5 0.1 155) 40px, oklch(0.5 0.1 155) 41px)",
          }}
        />
        <div className="max-w-4xl mx-auto flex flex-wrap items-center gap-3 relative z-10">
          <div className="w-12 h-12 rounded-xl bg-primary/10 border border-primary/30 flex items-center justify-center text-2xl shadow-gold">
            🏏
          </div>
          <div className="flex-1 min-w-0">
            <h1 className="text-2xl md:text-3xl font-display font-black text-primary tracking-tight leading-none">
              AI Gully Cricket Umpire
            </h1>
            <p className="text-xs text-muted-foreground mt-0.5 font-body">
              MoveNet Thunder · Pose detection · Knee-angle analysis
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2 ml-auto">
            {/* Model cache indicator */}
            <div
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-bold border ${
                modelCached
                  ? "bg-green-950/60 border-green-500/40 text-green-400"
                  : "bg-muted/40 border-border text-muted-foreground"
              }`}
            >
              <span>{modelCached ? "⚡" : "🔄"}</span>
              <span>{modelCached ? "Model Cached" : "Loading from CDN"}</span>
            </div>

            {/* Tournament button */}
            <button
              type="button"
              data-ocid="tournament.open_modal_button"
              onClick={() => setTournamentOpen(true)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-bold border bg-primary/10 border-primary/30 text-primary hover:bg-primary/20 transition-colors"
            >
              <Trophy size={14} />
              Tournament
            </button>

            {isRunning && (
              <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                className="flex items-center gap-2 bg-red-950/60 border border-red-500/50 rounded-full px-3 py-1"
              >
                <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                <span className="text-red-400 text-xs font-bold uppercase tracking-wider">
                  LIVE
                </span>
              </motion.div>
            )}
          </div>
        </div>

        {/* Active tournament chip */}
        {activeTournament && (
          <div className="max-w-4xl mx-auto mt-2 relative z-10">
            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-xl bg-primary/10 border border-primary/30 text-sm">
              <Trophy size={13} className="text-primary" />
              <span className="font-bold text-primary">
                {activeTournament.teamA} vs {activeTournament.teamB}
              </span>
              <span className="text-muted-foreground">|</span>
              <span className="text-foreground">
                Bowler: {activeTournament.bowler}
              </span>
              <button
                type="button"
                onClick={() => {
                  setActiveTournament(null);
                  activeTournamentRef.current = null;
                }}
                className="ml-1 text-muted-foreground hover:text-foreground transition-colors"
              >
                <X size={12} />
              </button>
            </div>
          </div>
        )}
      </header>

      <main className="flex-1 max-w-4xl mx-auto w-full px-4 py-5 flex flex-col gap-4">
        {/* ── Video / Canvas Area ── */}
        <div
          ref={stageRef}
          data-ocid="video.canvas_target"
          className="relative bg-card rounded-2xl overflow-hidden border border-border select-none"
          style={{ aspectRatio: "16/9" }}
          onPointerMove={handleStagePointerMove}
          onPointerUp={handleStagePointerUp}
        >
          <video
            ref={videoRef}
            className="w-full h-full object-cover"
            style={{ transform: isMirrored ? "scaleX(-1)" : "none" }}
            muted
            autoPlay
            playsInline
          />
          <canvas
            ref={canvasRef}
            className="absolute inset-0 w-full h-full pointer-events-none"
            style={{ transform: isMirrored ? "scaleX(-1)" : "none" }}
          />

          {/* Crease line drag handle (HTML overlay on top of canvas) */}
          {isRunning && (
            <div
              className="absolute left-0 right-0 z-10"
              style={{ top: `${creaseLineY}%`, transform: "translateY(-50%)" }}
            >
              {/* Invisible wider hit area for drag */}
              <div
                className="h-6 cursor-ns-resize flex items-center justify-end pr-2"
                onPointerDown={handleCreasePointerDown}
              >
                <span className="text-[10px] font-bold text-red-400/80 bg-black/60 px-1.5 py-0.5 rounded">
                  ↕ CREASE
                </span>
              </div>
            </div>
          )}

          {/* Camera swap button — top right */}
          {(isRunning || status === "ready" || status === "stopped") && (
            <motion.button
              data-ocid="camera.toggle"
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              whileTap={{ scale: 0.9 }}
              onClick={swapCamera}
              className="absolute top-3 right-3 z-20 flex items-center gap-1.5 px-3 py-2 rounded-xl bg-black/70 backdrop-blur-sm border border-white/20 text-white hover:bg-black/90 transition-colors text-xs font-bold shadow-lg"
              title={`Switch to ${
                facingMode === "environment" ? "front" : "back"
              } camera`}
            >
              <FlipHorizontal2 size={16} />
              {facingMode === "environment" ? "Back" : "Front"}
            </motion.button>
          )}

          {/* Corner angle overlay while running */}
          {isRunning && kneeAngle !== null && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="absolute top-3 left-3 z-10 bg-black/70 backdrop-blur-sm border border-white/10 rounded-xl px-3 py-2"
            >
              <div className="text-xs text-muted-foreground uppercase tracking-widest">
                Knee
              </div>
              <div
                className={`text-2xl font-display font-black leading-none ${
                  kneeAngle < 160 ? "text-red-400" : "text-green-400"
                }`}
              >
                {kneeAngle}°
              </div>
            </motion.div>
          )}

          {/* Foot over crease warning */}
          <AnimatePresence>
            {footOverCrease && isRunning && (
              <motion.div
                initial={{ opacity: 0, y: -8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className="absolute top-3 left-1/2 -translate-x-1/2 z-10 flex items-center gap-2 bg-red-950/90 border border-red-500/60 text-red-300 text-xs font-bold px-3 py-1.5 rounded-full"
              >
                ⚠️ FOOT OVER CREASE!
              </motion.div>
            )}
          </AnimatePresence>

          {/* Ball released flash */}
          <AnimatePresence>
            {ballReleasedFlash && (
              <motion.div
                initial={{ opacity: 0, scale: 1.15 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                className="absolute inset-0 flex items-center justify-center z-10 pointer-events-none"
              >
                <div className="text-3xl font-display font-black text-yellow-300 uppercase tracking-widest animate-glow bg-black/60 px-6 py-3 rounded-2xl border border-yellow-400/50">
                  🎯 BALL RELEASED!
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Idle / loading overlay */}
          <AnimatePresence>
            {!isRunning && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="absolute inset-0 flex flex-col items-center justify-center gap-4"
                style={{ background: "oklch(0.10 0.025 155 / 0.85)" }}
              >
                {isLoading ? (
                  <>
                    <div className="relative">
                      <div className="w-16 h-16 rounded-full border-4 border-primary/20 border-t-primary animate-spin" />
                      <div className="absolute inset-0 flex items-center justify-center text-2xl">
                        🧠
                      </div>
                    </div>
                    <div className="text-center">
                      <p className="text-primary font-bold text-lg">
                        Loading AI Model
                      </p>
                      <p className="text-muted-foreground text-sm">
                        Initialising TensorFlow.js MoveNet THUNDER...
                      </p>
                    </div>
                  </>
                ) : status === "error" ? (
                  <>
                    <div className="text-5xl">⚠️</div>
                    <div className="text-center px-6">
                      <p className="text-red-400 font-bold text-lg">Error</p>
                      <p className="text-muted-foreground text-sm mt-1">
                        {errorMsg}
                      </p>
                    </div>
                    <button
                      type="button"
                      onClick={retryLoad}
                      className="px-4 py-2 bg-primary text-primary-foreground rounded-lg text-sm font-bold hover:opacity-90 transition-opacity"
                    >
                      Retry
                    </button>
                  </>
                ) : (
                  <>
                    <div className="text-6xl">🏏</div>
                    <div className="text-center">
                      <p className="text-foreground font-bold text-lg">
                        {status === "stopped"
                          ? "Camera Stopped"
                          : "Ready to Analyse"}
                      </p>
                      <p className="text-muted-foreground text-sm">
                        {status === "stopped"
                          ? "Click Start Camera to resume"
                          : "Click Start Camera to begin pose detection"}
                      </p>
                    </div>
                  </>
                )}
              </motion.div>
            )}
          </AnimatePresence>

          {/* Scan line effect when running */}
          {isRunning && (
            <div
              className="absolute inset-x-0 h-0.5 pointer-events-none animate-scan"
              style={{
                background:
                  "linear-gradient(90deg, transparent, oklch(0.84 0.20 175 / 0.6), transparent)",
              }}
            />
          )}
        </div>

        {/* ── Delivery Counter Stats Row ── */}
        <div className="grid grid-cols-3 gap-3">
          <div
            data-ocid="stats.total.card"
            className="bg-card rounded-xl p-3 border border-border text-center"
          >
            <div className="text-xs text-muted-foreground uppercase tracking-widest mb-1">
              Total
            </div>
            <div className="text-3xl font-display font-black text-foreground">
              {totalBalls}
            </div>
            <div className="text-xs text-muted-foreground">Balls</div>
          </div>
          <div
            data-ocid="stats.legal.card"
            className="bg-card rounded-xl p-3 border border-green-500/20 text-center"
          >
            <div className="text-xs text-green-400/70 uppercase tracking-widest mb-1">
              Legal
            </div>
            <div className="text-3xl font-display font-black text-green-400">
              {legalBalls}
            </div>
            <div className="text-xs text-muted-foreground">Balls</div>
          </div>
          <div
            data-ocid="stats.noballs.card"
            className="bg-card rounded-xl p-3 border border-red-500/20 text-center"
          >
            <div className="text-xs text-red-400/70 uppercase tracking-widest mb-1">
              No Balls
            </div>
            <div className="text-3xl font-display font-black text-red-400">
              {noBalls}
            </div>
            <div className="text-xs text-muted-foreground">Balls</div>
          </div>
        </div>

        {/* ── Locked Verdict Banner ── */}
        <AnimatePresence>
          {lockedVerdict && (
            <motion.div
              key="locked-verdict"
              initial={{ opacity: 0, y: -12, scale: 0.96 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -8, scale: 0.97 }}
              transition={{
                duration: 0.25,
                type: "spring",
                stiffness: 260,
                damping: 22,
              }}
              data-ocid="verdict.card"
              className={`rounded-2xl p-5 border-2 ${
                lockedVerdict.verdict === "noball"
                  ? "border-red-500 animate-no-ball"
                  : "border-green-500"
              }`}
              style={{
                background:
                  lockedVerdict.verdict === "noball"
                    ? "oklch(0.20 0.08 25 / 0.7)"
                    : "oklch(0.18 0.06 155 / 0.7)",
              }}
            >
              <div className="flex items-start justify-between gap-3">
                <div className="flex-1">
                  <div
                    className={`text-xs font-bold uppercase tracking-[0.2em] mb-1 ${
                      lockedVerdict.verdict === "noball"
                        ? "text-red-400/70"
                        : "text-green-400/70"
                    }`}
                  >
                    🔒 DECISION
                  </div>
                  <div
                    className={`text-4xl md:text-6xl font-display font-black tracking-widest ${
                      lockedVerdict.verdict === "noball"
                        ? "text-red-400 animate-glow"
                        : "text-green-400"
                    }`}
                  >
                    {lockedVerdict.verdict === "noball"
                      ? "⚡ NO BALL"
                      : "✅ LEGAL BALL"}
                  </div>
                  <div
                    className={`text-sm mt-1.5 flex flex-wrap items-center gap-3 ${
                      lockedVerdict.verdict === "noball"
                        ? "text-red-300/70"
                        : "text-green-300/70"
                    }`}
                  >
                    <span>
                      Locked at delivery · Angle: {lockedVerdict.angle}°
                    </span>
                    <span className="font-mono text-xs opacity-80">
                      {lockedVerdict.timestamp}
                    </span>
                    {/* Confidence badge */}
                    <span
                      className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-bold border ${
                        lockedVerdict.confidence >= 75
                          ? "bg-green-950/60 border-green-500/40 text-green-300"
                          : lockedVerdict.confidence >= 50
                            ? "bg-yellow-950/60 border-yellow-500/40 text-yellow-300"
                            : "bg-red-950/60 border-red-500/40 text-red-300"
                      }`}
                    >
                      🎯 {lockedVerdict.confidence}% confidence
                    </span>
                  </div>

                  {/* Clip / Replay actions */}
                  <div className="flex gap-2 mt-3 flex-wrap">
                    {lastClipUrl && (
                      <>
                        <button
                          type="button"
                          data-ocid="clip.open_modal_button"
                          onClick={() => setShowClipModal(true)}
                          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white/10 border border-white/15 text-white text-xs font-bold hover:bg-white/15 transition-colors"
                        >
                          <Play size={13} /> Replay Clip
                        </button>
                        <a
                          href={lastClipUrl}
                          download="delivery-clip.webm"
                          data-ocid="clip.download_button"
                          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-primary/20 border border-primary/30 text-primary text-xs font-bold hover:bg-primary/30 transition-colors"
                        >
                          <Download size={13} /> Download
                        </a>
                      </>
                    )}
                    {replayFrames.length > 0 && (
                      <button
                        type="button"
                        data-ocid="replay.open_modal_button"
                        onClick={() => setShowReplay(true)}
                        className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-accent/10 border border-accent/20 text-accent text-xs font-bold hover:bg-accent/15 transition-colors"
                      >
                        🎬 Slow-Mo Replay
                      </button>
                    )}
                  </div>
                </div>
                <button
                  type="button"
                  data-ocid="verdict.close_button"
                  onClick={() => setLockedVerdict(null)}
                  className="mt-1 p-1.5 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 transition-colors text-white/60 hover:text-white/90"
                  title="Clear decision"
                >
                  <X size={16} />
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* ── Live Verdict Banner (when not locked) ── */}
        <AnimatePresence mode="wait">
          {!lockedVerdict && verdict === "noball" && (
            <motion.div
              key="noball"
              data-ocid="verdict.panel"
              initial={{ opacity: 0, y: 10, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -10, scale: 0.95 }}
              transition={{ duration: 0.2 }}
              className="rounded-2xl p-6 text-center border-2 border-red-500 animate-no-ball"
              style={{ background: "oklch(0.20 0.08 25 / 0.6)" }}
            >
              <div className="text-5xl md:text-7xl font-display font-black text-red-400 tracking-widest animate-glow">
                ⚡ NO BALL
              </div>
              <div className="text-red-300/80 text-sm mt-2">
                Knee bent beyond legal limit · Angle: {kneeAngle}° (must be ≥
                160°)
              </div>
            </motion.div>
          )}

          {!lockedVerdict && verdict === "legal" && (
            <motion.div
              key="legal"
              data-ocid="verdict.panel"
              initial={{ opacity: 0, y: 10, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -10, scale: 0.95 }}
              transition={{ duration: 0.2 }}
              className="rounded-2xl p-6 text-center border-2 border-green-500"
              style={{ background: "oklch(0.18 0.06 155 / 0.6)" }}
            >
              <div className="text-5xl md:text-7xl font-display font-black text-green-400 tracking-widest">
                ✅ LEGAL BALL
              </div>
              <div className="text-green-300/80 text-sm mt-2">
                Bowling action within legal limits · Angle: {kneeAngle}°
              </div>
            </motion.div>
          )}

          {!lockedVerdict && verdict === "detecting" && (
            <motion.div
              key="detecting"
              data-ocid="verdict.panel"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="rounded-2xl p-5 text-center border border-border bg-card"
            >
              <div className="text-2xl font-display font-bold text-muted-foreground animate-pulse">
                🔍 Detecting bowler position...
              </div>
              <div className="text-muted-foreground/60 text-sm mt-1">
                Stand in frame — position your full body for detection
              </div>
            </motion.div>
          )}

          {!lockedVerdict && !verdict && (
            <motion.div
              key="idle"
              data-ocid="verdict.panel"
              className="rounded-2xl p-5 text-center border border-border bg-card"
            >
              <div className="text-xl font-display font-semibold text-muted-foreground">
                Waiting for camera feed
              </div>
              <div className="text-muted-foreground/60 text-sm mt-1">
                Start the camera to begin umpire analysis
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* ── Stability Bar ── */}
        {isRunning && (
          <motion.div
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-card rounded-xl border border-border px-4 py-3 flex items-center gap-3"
          >
            <div className="text-xs text-muted-foreground uppercase tracking-widest whitespace-nowrap">
              Pose Stability
            </div>
            <div className="flex gap-1.5 flex-1">
              {STABILITY_DOTS.map((dotKey, i) => (
                <div
                  key={dotKey}
                  className={`flex-1 h-2.5 rounded-full transition-all duration-300 ${
                    i < stabilityCount
                      ? isStable
                        ? "bg-green-400 shadow-[0_0_6px_rgba(74,222,128,0.6)]"
                        : "bg-primary/80"
                      : "bg-muted/40"
                  }`}
                />
              ))}
            </div>
            <div
              className={`text-xs font-bold whitespace-nowrap ${
                isStable ? "text-green-400" : "text-muted-foreground"
              }`}
            >
              {isStable
                ? "STABLE"
                : `${stabilityCount}/${STABILITY_MIN_READINGS}`}
            </div>
          </motion.div>
        )}

        {/* ── Knee Angle & Legend Panel ── */}
        <div className="grid grid-cols-3 gap-3">
          <div
            data-ocid="angle.panel"
            className="col-span-2 bg-card rounded-xl p-4 border border-border flex flex-col items-center justify-center"
          >
            <div className="text-xs text-muted-foreground uppercase tracking-widest mb-1">
              Knee Angle
            </div>
            <div
              className={`text-5xl font-display font-black transition-colors duration-200 ${
                kneeAngle === null
                  ? "text-muted-foreground"
                  : kneeAngle < 160
                    ? "text-red-400"
                    : "text-green-400"
              }`}
            >
              {kneeAngle !== null ? `${kneeAngle}°` : "—"}
            </div>
            <div className="text-xs text-muted-foreground mt-1">
              Legal threshold ≥ 160°
            </div>
          </div>

          <div className="bg-card rounded-xl p-4 border border-border flex flex-col gap-2">
            <div className="text-xs text-muted-foreground uppercase tracking-widest">
              Legend
            </div>
            <div className="flex items-center gap-2">
              <div
                className="w-3 h-3 rounded-full"
                style={{ background: "#fbbf24" }}
              />
              <span className="text-xs text-foreground">Leg joints</span>
            </div>
            <div className="flex items-center gap-2">
              <div
                className="w-3 h-3 rounded-full"
                style={{ background: "#22d3ee" }}
              />
              <span className="text-xs text-foreground">Upper body</span>
            </div>
            <div className="flex items-center gap-2">
              <div
                className="w-3 h-3 rounded-full"
                style={{ background: "#f97316" }}
              />
              <span className="text-xs text-foreground">Knee point</span>
            </div>
            <div className="flex items-center gap-2">
              <div
                className="w-3 h-0.5 rounded"
                style={{ background: "#ff3c3c", border: "1px dashed #ff3c3c" }}
              />
              <span className="text-xs text-foreground">Crease line</span>
            </div>
          </div>
        </div>

        {/* ── Tournament Scoreboard ── */}
        {activeTournament && (
          <motion.div
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            data-ocid="tournament.panel"
            className="bg-card rounded-xl border border-primary/20 p-4"
          >
            <div className="flex items-center gap-2 mb-3">
              <Trophy size={16} className="text-primary" />
              <span className="text-sm font-display font-bold text-primary uppercase tracking-wider">
                Match Scoreboard
              </span>
            </div>
            <div className="grid grid-cols-4 gap-2 text-center text-xs">
              <div className="text-muted-foreground font-bold uppercase">
                Bowler
              </div>
              <div className="text-muted-foreground font-bold uppercase">
                Total
              </div>
              <div className="text-green-400/80 font-bold uppercase">Legal</div>
              <div className="text-red-400/80 font-bold uppercase">
                No Balls
              </div>
              <div className="font-bold text-foreground truncate">
                {activeTournament.bowler}
              </div>
              <div className="font-display font-black text-foreground text-lg">
                {currentBowlerStats?.total ?? 0}
              </div>
              <div className="font-display font-black text-green-400 text-lg">
                {currentBowlerStats?.legal ?? 0}
              </div>
              <div className="font-display font-black text-red-400 text-lg">
                {currentBowlerStats?.noBall ?? 0}
              </div>
            </div>
          </motion.div>
        )}

        {/* ── Control Buttons ── */}
        <div className="flex gap-3">
          <motion.button
            data-ocid="camera.primary_button"
            whileTap={{ scale: 0.97 }}
            onClick={() => startCamera()}
            disabled={isRunning || isLoading || !tfReady}
            className="flex-1 py-3.5 px-6 rounded-xl font-display font-bold text-sm uppercase tracking-widest bg-primary text-primary-foreground disabled:opacity-40 disabled:cursor-not-allowed hover:opacity-90 transition-opacity shadow-gold"
          >
            {isLoading ? (
              <span className="flex items-center justify-center gap-2">
                <span className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin" />
                Loading Model...
              </span>
            ) : (
              "▶ Start Camera"
            )}
          </motion.button>

          <motion.button
            data-ocid="camera.secondary_button"
            whileTap={{ scale: 0.97 }}
            onClick={stopCamera}
            disabled={!isRunning}
            className="flex-1 py-3.5 px-6 rounded-xl font-display font-bold text-sm uppercase tracking-widest bg-secondary text-secondary-foreground disabled:opacity-40 disabled:cursor-not-allowed hover:opacity-90 transition-opacity border border-border"
          >
            ⏹ Stop Camera
          </motion.button>
        </div>

        {/* ── Status Bar ── */}
        <div
          data-ocid="status.panel"
          className="text-center text-sm py-2 px-4 rounded-lg bg-card border border-border"
        >
          {status === "error" ? (
            <span className="text-red-400" data-ocid="status.error_state">
              ❌ {errorMsg}
            </span>
          ) : statusMessage ? (
            <span
              className={`${
                status === "loading"
                  ? "text-primary animate-pulse"
                  : status === "ready"
                    ? "text-green-400"
                    : status === "stopped" || status === "idle"
                      ? "text-muted-foreground"
                      : lockedVerdict
                        ? "text-primary font-semibold"
                        : isStable
                          ? "text-accent"
                          : "text-muted-foreground"
              }`}
            >
              {statusMessage}
            </span>
          ) : null}
        </div>

        {/* ── How It Works ── */}
        <div className="bg-card rounded-xl border border-border p-4 grid grid-cols-2 md:grid-cols-5 gap-4 text-center">
          {[
            { icon: "📹", title: "Camera", desc: "Back camera default" },
            {
              icon: "🤖",
              title: "MoveNet Thunder",
              desc: "17 keypoints · full body gate",
            },
            { icon: "📐", title: "Knee Angle", desc: "Hip→Knee→Ankle ≥ 160°" },
            {
              icon: "🔒",
              title: "Decision Lock",
              desc: "Locked at minimum angle",
            },
            { icon: "🏆", title: "Tournament", desc: "Track bowler stats" },
          ].map((item) => (
            <div key={item.title} className="flex flex-col items-center gap-2">
              <span className="text-2xl">{item.icon}</span>
              <div className="text-xs font-display font-bold text-foreground">
                {item.title}
              </div>
              <div className="text-xs text-muted-foreground">{item.desc}</div>
            </div>
          ))}
        </div>
      </main>

      {/* ── Footer ── */}
      <footer className="py-4 text-center text-xs text-muted-foreground border-t border-border">
        © {new Date().getFullYear()} · Built with ❤️ using{" "}
        <a
          href={`https://caffeine.ai?utm_source=caffeine-footer&utm_medium=referral&utm_content=${encodeURIComponent(window.location.hostname)}`}
          target="_blank"
          rel="noopener noreferrer"
          className="text-primary hover:underline"
        >
          caffeine.ai
        </a>
        {" · "}
        <span className="text-muted-foreground/60">
          Powered by TensorFlow.js + MoveNet Thunder
        </span>
      </footer>
    </div>
  );
}
