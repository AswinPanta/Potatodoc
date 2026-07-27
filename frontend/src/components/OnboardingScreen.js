import React, { useState, useEffect, useCallback } from "react";
import Typography from "@material-ui/core/Typography";
import LinearProgress from "@material-ui/core/LinearProgress";
import Avatar from "@material-ui/core/Avatar";
import LocalFlorist from "@material-ui/icons/LocalFlorist";
import CheckCircle from "@material-ui/icons/CheckCircle";
import Error from "@material-ui/icons/Error";
import HourglassEmpty from "@material-ui/icons/HourglassEmpty";
import CloudDownload from "@material-ui/icons/CloudDownload";

const POLL_INTERVAL = 1500; // ms between status checks

const STATUS_API = "/setup/status";

const MODEL_ORDER = ["cnn_baseline", "transfer_learning", "mobilenetv2"];

const MODEL_DISPLAY = {
  cnn_baseline: { name: "CNN Baseline", icon: "🔬", size: "~2.5 MB" },
  transfer_learning: { name: "Transfer Learning", icon: "🧠", size: "~227 MB" },
  mobilenetv2: { name: "MobileNetV2", icon: "⚡", size: "~25 MB" },
};

function StatusIcon({ status }) {
  if (status === "done") return <CheckCircle style={{ color: "#2E7D32", fontSize: 20 }} />;
  if (status === "error") return <Error style={{ color: "#C62828", fontSize: 20 }} />;
  if (status === "downloading" || status === "extracting")
    return <CloudDownload style={{ color: "#1565C0", fontSize: 20 }} />;
  return <HourglassEmpty style={{ color: "#BDBDBD", fontSize: 20 }} />;
}

export default function OnboardingScreen({ onComplete }) {
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);
  const [dots, setDots] = useState("");

  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch(STATUS_API);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setStatus(data);
      setError(null);

      if (data.state === "complete") {
        setTimeout(() => onComplete(), 1200);
      }
    } catch (err) {
      setError(err.message);
    }
  }, [onComplete]);

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, POLL_INTERVAL);
    return () => clearInterval(interval);
  }, [fetchStatus]);

  // Animate dots
  useEffect(() => {
    const interval = setInterval(() => {
      setDots((d) => (d.length >= 3 ? "" : d + "."));
    }, 500);
    return () => clearInterval(interval);
  }, []);

  const state = status?.state || "idle";
  const overall = status?.overall_progress || 0;
  const models = status?.models || {};
  const isComplete = state === "complete";
  const isError = state === "error";

  // If models are already ready, skip straight through
  if (isComplete && status?.overall_progress === 100) {
    // onComplete is called via setTimeout above
  }

  return (
    <div style={styles.container}>
      {/* Animated background */}
      <div style={styles.bgPattern} />

      <div style={styles.card}>
        {/* Logo */}
        <div style={styles.logoWrap}>
          <Avatar style={styles.logoAvatar}>
            <LocalFlorist style={{ fontSize: 48, color: "#FFFFFF" }} />
          </Avatar>
        </div>

        <Typography style={styles.title}>PotatoDoc</Typography>
        <Typography style={styles.subtitle}>
          AI-Powered Potato Disease Detection
        </Typography>

        {/* Status message */}
        <div style={styles.statusBox}>
          {isError ? (
            <Typography style={styles.errorText}>
              <Error style={{ fontSize: 18, verticalAlign: "middle", marginRight: 8 }} />
              {status?.error || "Download failed"}
            </Typography>
          ) : isComplete ? (
            <Typography style={styles.successText}>
              <CheckCircle style={{ fontSize: 18, verticalAlign: "middle", marginRight: 8 }} />
              All models ready!
            </Typography>
          ) : (
            <Typography style={styles.statusText}>
              {status?.message || `Preparing${dots}`}
            </Typography>
          )}
        </div>

        {/* Overall progress bar */}
        {!isComplete && (
          <div style={styles.progressWrap}>
            <div style={styles.progressLabel}>
              <Typography style={styles.progressLabelText}>Overall Progress</Typography>
              <Typography style={styles.progressPercent}>{overall}%</Typography>
            </div>
            <LinearProgress
              variant="determinate"
              value={overall}
              style={styles.progressBar}
            />
          </div>
        )}

        {/* Per-model status */}
        <div style={styles.modelList}>
          {MODEL_ORDER.map((key, idx) => {
            const modelStatus = models[key] || "pending";
            const info = MODEL_DISPLAY[key];
            const isActive = modelStatus === "downloading" || modelStatus === "extracting";
            const isDone = modelStatus === "done";
            const isFailed = modelStatus === "error";

            return (
              <div
                key={key}
                style={{
                  ...styles.modelRow,
                  ...(isActive ? styles.modelRowActive : {}),
                  ...(isDone ? styles.modelRowDone : {}),
                }}
              >
                <div style={styles.modelLeft}>
                  <StatusIcon status={modelStatus} />
                  <div style={styles.modelInfo}>
                    <Typography style={styles.modelName}>
                      {info.icon} {info.name}
                    </Typography>
                    <Typography style={styles.modelSize}>{info.size}</Typography>
                  </div>
                </div>
                <div style={styles.modelRight}>
                  {isActive && (
                    <div style={styles.downloadingBadge}>
                      <CloudDownload style={{ fontSize: 14, marginRight: 4 }} />
                      {modelStatus === "extracting" ? "Extracting" : "Downloading"}
                      {dots}
                    </div>
                  )}
                  {isDone && (
                    <div style={styles.doneBadge}>
                      <CheckCircle style={{ fontSize: 14, marginRight: 4 }} />
                      Ready
                    </div>
                  )}
                  {isFailed && (
                    <div style={styles.errorBadge}>
                      <Error style={{ fontSize: 14, marginRight: 4 }} />
                      Failed
                    </div>
                  )}
                  {modelStatus === "pending" && !isActive && (
                    <div style={styles.pendingBadge}>Waiting{dots}</div>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        {/* Tip */}
        <div style={styles.tip}>
          <Typography style={styles.tipText}>
            Models are downloaded once and cached locally for fast startup.
          </Typography>
        </div>
      </div>
    </div>
  );
}

const styles = {
  container: {
    minHeight: "100vh",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    background: "linear-gradient(135deg, #1B5E20 0%, #2E7D32 40%, #388E3C 100%)",
    position: "relative",
    overflow: "hidden",
    fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
  },
  bgPattern: {
    position: "absolute",
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundImage:
      "radial-gradient(circle at 20% 80%, rgba(255,255,255,0.06) 0%, transparent 50%), " +
      "radial-gradient(circle at 80% 20%, rgba(255,255,255,0.04) 0%, transparent 50%)",
    pointerEvents: "none",
  },
  card: {
    background: "#FFFFFF",
    borderRadius: 28,
    padding: "40px 36px 32px",
    maxWidth: 440,
    width: "90%",
    boxShadow: "0 20px 60px rgba(0,0,0,0.2)",
    position: "relative",
    zIndex: 1,
  },
  logoWrap: {
    display: "flex",
    justifyContent: "center",
    marginBottom: 16,
  },
  logoAvatar: {
    width: 72,
    height: 72,
    background: "linear-gradient(135deg, #2E7D32, #4CAF50)",
    boxShadow: "0 8px 24px rgba(46,125,50,0.3)",
  },
  title: {
    fontSize: 32,
    fontWeight: 800,
    textAlign: "center",
    color: "#1B5E20",
    letterSpacing: "-0.5px",
    marginBottom: 4,
  },
  subtitle: {
    fontSize: 14,
    textAlign: "center",
    color: "#757575",
    marginBottom: 24,
    fontWeight: 500,
  },
  statusBox: {
    textAlign: "center",
    padding: "12px 16px",
    borderRadius: 12,
    backgroundColor: "#F1F8E9",
    marginBottom: 20,
  },
  statusText: {
    fontSize: 14,
    color: "#2E7D32",
    fontWeight: 600,
  },
  successText: {
    fontSize: 14,
    color: "#2E7D32",
    fontWeight: 700,
  },
  errorText: {
    fontSize: 14,
    color: "#C62828",
    fontWeight: 600,
  },
  progressWrap: {
    marginBottom: 20,
  },
  progressLabel: {
    display: "flex",
    justifyContent: "space-between",
    marginBottom: 6,
  },
  progressLabelText: {
    fontSize: 12,
    fontWeight: 600,
    color: "#757575",
    textTransform: "uppercase",
    letterSpacing: "0.5px",
  },
  progressPercent: {
    fontSize: 12,
    fontWeight: 700,
    color: "#2E7D32",
  },
  progressBar: {
    height: 8,
    borderRadius: 4,
    backgroundColor: "#E8F5E9",
    overflow: "hidden",
  },
  progressFill: {
    backgroundColor: "linear-gradient(90deg, #2E7D32, #4CAF50)",
    borderRadius: 4,
    transition: "width 0.5s ease",
  },
  modelList: {
    display: "flex",
    flexDirection: "column",
    gap: 8,
    marginBottom: 16,
  },
  modelRow: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "12px 14px",
    borderRadius: 12,
    backgroundColor: "#FAFAFA",
    border: "1px solid #F0F0F0",
    transition: "all 0.3s ease",
  },
  modelRowActive: {
    backgroundColor: "#E8F5E9",
    border: "1px solid #C8E6C9",
    boxShadow: "0 2px 8px rgba(46,125,50,0.1)",
  },
  modelRowDone: {
    backgroundColor: "#F1F8E9",
    border: "1px solid #C8E6C9",
  },
  modelLeft: {
    display: "flex",
    alignItems: "center",
    gap: 10,
  },
  modelInfo: {
    display: "flex",
    flexDirection: "column",
  },
  modelName: {
    fontSize: 14,
    fontWeight: 600,
    color: "#1B1B1B",
  },
  modelSize: {
    fontSize: 11,
    color: "#9E9E9E",
    fontWeight: 500,
  },
  modelRight: {
    display: "flex",
    alignItems: "center",
  },
  downloadingBadge: {
    display: "flex",
    alignItems: "center",
    fontSize: 12,
    fontWeight: 600,
    color: "#1565C0",
    backgroundColor: "#E3F2FD",
    padding: "4px 10px",
    borderRadius: 8,
  },
  doneBadge: {
    display: "flex",
    alignItems: "center",
    fontSize: 12,
    fontWeight: 600,
    color: "#2E7D32",
    backgroundColor: "#E8F5E9",
    padding: "4px 10px",
    borderRadius: 8,
  },
  errorBadge: {
    display: "flex",
    alignItems: "center",
    fontSize: 12,
    fontWeight: 600,
    color: "#C62828",
    backgroundColor: "#FFEBEE",
    padding: "4px 10px",
    borderRadius: 8,
  },
  pendingBadge: {
    fontSize: 12,
    fontWeight: 500,
    color: "#9E9E9E",
    padding: "4px 10px",
  },
  tip: {
    textAlign: "center",
    padding: "8px 12px",
    borderRadius: 8,
    backgroundColor: "#FFF8E1",
  },
  tipText: {
    fontSize: 11,
    color: "#F57F17",
    fontWeight: 500,
  },
};
