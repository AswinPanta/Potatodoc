import { useState, useEffect } from "react";
import axios from "axios";

const API_BASE = process.env.REACT_APP_API_URL || "http://localhost:8000";
const TIMEOUT_MS = 30000; // 30s timeout for slow model inference

export function useModels() {
  const [models, setModels] = useState(["cnn-baseline", "transfer-learning", "mobilenetv2", "ensemble"]);
  const [modelNames, setModelNames] = useState({
    "cnn-baseline": "CNN Baseline",
    "transfer-learning": "Transfer Learning",
    "mobilenetv2": "MobileNetV2",
  });

  useEffect(() => {
    const fetchModels = async () => {
      try {
        const res = await axios.get(`${API_BASE}/models`);
        setModels([...res.data.models, "ensemble"]);
        setModelNames(res.data.modelNames);
      } catch (err) {
        console.error("Failed to fetch models:", err);
      }
    };
    fetchModels();
  }, []);

  return { models, modelNames };
}

export function usePrediction() {
  const [isLoading, setIsloading] = useState(false);
  const [data, setData] = useState(null);

  const sendFile = async (selectedFile, selectedModel) => {
    if (!selectedFile) return;
    setIsloading(true);
    let formData = new FormData();
    formData.append("file", selectedFile);
    try {
      let res = await axios({
        method: "post",
        url: `${API_BASE}/predict?model_id=${selectedModel}`,
        data: formData,
        timeout: TIMEOUT_MS,
      });
      if (res.status === 200) {
        setData(res.data);
        setIsloading(false);
        return res.data;
      }
    } catch (err) {
      console.error("Prediction failed:", err);
    }
    setIsloading(false);
    return null;
  };

  return { data, setData, isLoading, sendFile };
}

/**
 * Fetch Grad-CAM heatmap overlay from the API.
 * Returns heatmap data for the given model, or all models if "ensemble".
 */
export async function fetchGradcam(file, modelId) {
  if (!file) return null;
  let formData = new FormData();
  formData.append("file", file);
    try {
      let res = await axios({
        method: "post",
        url: `${API_BASE}/gradcam?model_id=${modelId}`,
        data: formData,
        timeout: TIMEOUT_MS,
      });
    if (res.status === 200 && !res.data.error) {
      return res.data;
    }
    return null;
  } catch (err) {
    console.error("Grad-CAM failed:", err);
    return null;
  }
}
