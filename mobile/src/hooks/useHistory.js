import { useState, useEffect } from "react";
import AsyncStorage from "@react-native-async-storage/async-storage";

const STORAGE_KEY = "potatoDocHistory";

export function useHistory() {
  const [history, setHistory] = useState([]);

  useEffect(() => {
    (async () => {
      try {
        const saved = await AsyncStorage.getItem(STORAGE_KEY);
        if (saved) {
          setHistory(JSON.parse(saved));
        }
      } catch (err) {
        console.error("Failed to load history:", err);
      }
    })();
  }, []);

  const saveToHistory = async (newPrediction) => {
    const newHistory = [
      {
        id: Date.now(),
        ...newPrediction,
        timestamp: new Date().toLocaleString(),
      },
      ...history,
    ].slice(0, 20);
    setHistory(newHistory);
    try {
      await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(newHistory));
    } catch (err) {
      console.error("Failed to save history:", err);
    }
  };

  return { history, saveToHistory };
}
