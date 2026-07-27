import { useState, useEffect } from "react";

export function useHistory() {
  const [history, setHistory] = useState([]);

  useEffect(() => {
    try {
      const savedHistory = localStorage.getItem("potatoDocHistory");
      if (savedHistory) {
        setHistory(JSON.parse(savedHistory));
      }
    } catch {
      localStorage.removeItem("potatoDocHistory");
    }
  }, []);

  const saveToHistory = (newPrediction) => {
    const entry = {
      id: Date.now(),
      ...newPrediction,
      timestamp: new Date().toLocaleString(),
    };
    // Strip large heatmap base64 data to avoid exceeding localStorage quota
    if (entry.heatmaps) {
      entry._hasHeatmaps = true;
      delete entry.heatmaps;
    }
    const newHistory = [entry, ...history].slice(0, 20);
    setHistory(newHistory);
    try {
      localStorage.setItem("potatoDocHistory", JSON.stringify(newHistory));
    } catch {
      // Storage full — try with fewer entries
      const reduced = newHistory.slice(0, 5);
      try {
        localStorage.setItem("potatoDocHistory", JSON.stringify(reduced));
        setHistory(reduced);
      } catch {
        // Give up silently
      }
    }
  };

  return { history, saveToHistory };
}
