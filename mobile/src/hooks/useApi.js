import { useState, useEffect, useRef } from "react";
import axios from "axios";
import { File, UploadType } from "expo-file-system";

const API_BASE = process.env.EXPO_PUBLIC_API_URL || "http://localhost:8000";

export const MODEL_IDS = ["cnn-baseline", "transfer-learning", "mobilenetv2"];

// ---- Helpers -----------------------------------------------------------

/** Quick GET /ping to verify the server is reachable. */
async function checkConnectivity() {
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 10000);
    const res = await fetch(`${API_BASE}/ping`, { signal: controller.signal });
    clearTimeout(timer);
    const text = await res.text();
    console.log("[Connectivity] /ping responded:", res.status, text);
    return true;
  } catch (err) {
    console.warn("[Connectivity] /ping failed:", err.message || err);
    return false;
  }
}

/**
 * Upload a local file using expo-file-system's native upload API.
 * This uses the platform's native networking stack, bypassing React
 * Native's broken FormData / Blob implementation entirely.
 */
async function uploadNative(fileUri, endpointPath, queryParams = "") {
  const TIMEOUT_MS = 30000; // 30-second timeout

  try {
    const file = new File(fileUri);

    // Race the upload against a timeout
    const uploadPromise = file.upload(
      `${API_BASE}${endpointPath}${queryParams}`,
      {
        uploadType: UploadType.MULTIPART,
        fieldName: "file",
        httpMethod: "POST",
      }
    );

    const result = await Promise.race([
      uploadPromise,
      new Promise((_, reject) =>
        setTimeout(() => reject(new Error("Request timed out")), TIMEOUT_MS)
      ),
    ]);

    if (result.status !== 200) {
      console.warn("[uploadNative] HTTP", result.status);
      return null;
    }
    return JSON.parse(result.body);
  } catch (err) {
    console.warn("[uploadNative] failed:", err.message || err);
    return null;
  }
}

// ---- Hooks -------------------------------------------------------------

/**
 * useWakeUp — two-phase startup:
 *   1. Ping the Space to wake it from cold sleep (retry up to 3×).
 *   2. Send the bundled icon to each model endpoint in parallel so their
 *      weights are loaded into memory before the user snaps a photo.
 */
export function useWakeUp() {
  const [awake, setAwake] = useState(false);
  const [warming, setWarming] = useState(true);
  const [loadingModels, setLoadingModels] = useState([]);
  const [statusText, setStatusText] = useState("Connecting to server...");
  const attempts = useRef(0);
  const warmedRef = useRef(false);

  useEffect(() => {
    let cancelled = false;

    const run = async () => {
      // ---------- Phase 1: Wake the Space ----------
      while (attempts.current < 3 && !cancelled) {
        attempts.current += 1;
        try {
          const res = await axios.get(`${API_BASE}/ping`, {
            timeout: 15000,
          });
          if (!cancelled && res.status === 200) {
            setAwake(true);
            break;
          }
        } catch (_) {
          // cold-starting, retry
        }
        if (!cancelled && attempts.current < 3) {
          setStatusText("Retrying connection...");
          await new Promise((r) => setTimeout(r, 2000));
        }
      }

      if (cancelled || warmedRef.current) return;
      warmedRef.current = true;

      // ---------- Phase 2: Warm models in parallel ----------
      setStatusText("Loading models...");

      // Resolve the bundled icon to a local file URI
      const { Image } = require("react-native");
      const assetSource = Image.resolveAssetSource(
        require("../../assets/icon.png")
      );
      const warmupUri = assetSource.uri;

      setLoadingModels([...MODEL_IDS]);

      await Promise.all(
        MODEL_IDS.map(async (modelId) => {
          if (cancelled) return;
          try {
            const file = new File(warmupUri);
            await file.upload(
              `${API_BASE}/predict?model_id=${modelId}`,
              {
                uploadType: UploadType.MULTIPART,
                fieldName: "file",
                httpMethod: "POST",
              }
            );
          } catch (_) {
            // non-fatal
          }
          if (!cancelled) {
            setLoadingModels((prev) => prev.filter((m) => m !== modelId));
          }
        })
      );

      if (!cancelled) {
        setStatusText("Ready");
        setWarming(false);
      }
    };

    run();
    return () => {
      cancelled = true;
    };
  }, []);

  return { awake, warming, loadingModels, statusText };
}

export function useModels() {
  const [models, setModels] = useState([
    "cnn-baseline",
    "transfer-learning",
    "mobilenetv2",
    "ensemble",
  ]);
  const [modelNames, setModelNames] = useState({
    "cnn-baseline": "CNN Baseline",
    "transfer-learning": "Transfer Learning",
    "mobilenetv2": "MobileNetV2",
  });

  useEffect(() => {
    const fetchModels = async () => {
      try {
        const res = await axios.get(`${API_BASE}/models`, { timeout: 10000 });
        setModels([...res.data.models, "ensemble"]);
        setModelNames(res.data.modelNames);
      } catch (err) {
        console.error("[Models fetch]", err.message || err);
      }
    };
    fetchModels();
  }, []);

  return { models, modelNames };
}

export function usePrediction() {
  const [isLoading, setIsloading] = useState(false);
  const [data, setData] = useState(null);
  const checkedOk = useRef(false);

  const sendFile = async (selectedFile, selectedModel) => {
    if (!selectedFile || !selectedFile.uri) return null;

    // ---- Diagnostic: test basic connectivity once (cache result) ----
    if (!checkedOk.current) {
      const ok = await checkConnectivity();
      checkedOk.current = ok;
      if (!ok) {
        console.error("[Prediction] Cannot reach API — /ping failed");
        return null;
      }
    }

    setIsloading(true);
    const result = await uploadNative(
      selectedFile.uri,
      "/predict",
      `?model_id=${selectedModel}`
    );

    if (result) {
      setData(result);
    }
    setIsloading(false);
    return result;
  };

  return { data, setData, isLoading, sendFile };
}

/**
 * Fetch Grad-CAM heatmap overlay from the API.
 * Uses the same native upload approach.
 */
export async function fetchGradcam(file, modelId) {
  if (!file || !file.uri) return null;

  try {
    const result = await uploadNative(
      file.uri,
      "/gradcam",
      `?model_id=${modelId}`
    );
    return result;
  } catch (err) {
    console.warn("[fetchGradcam] failed:", err.message || err);
    return null;
  }
}
