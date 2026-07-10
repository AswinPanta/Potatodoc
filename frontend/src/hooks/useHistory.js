import { useState, useEffect } from "react";

export function useHistory() {
  const [history, setHistory] = useState([]);

  useEffect(() => {
    const savedHistory = localStorage.getItem("potatoDocHistory");
    if (savedHistory) {
      setHistory(JSON.parse(savedHistory));
    }
  }, []);

  const saveToHistory = (newPrediction) => {
    const newHistory = [
      {
        id: Date.now(),
        ...newPrediction,
        timestamp: new Date().toLocaleString(),
      },
      ...history,
    ].slice(0, 20);
    setHistory(newHistory);
    localStorage.setItem("potatoDocHistory", JSON.stringify(newHistory));
  };

  return { history, saveToHistory };
}
