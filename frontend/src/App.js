import React, { useState, useEffect } from "react";
import HomePage from "./pages/HomePage";
import OnboardingScreen from "./components/OnboardingScreen";

const STATUS_API = "/setup/status";

function App() {
  const [setupComplete, setSetupComplete] = useState(false);
  const [checking, setChecking] = useState(true);

  useEffect(() => {
    // Quick check: if models are already ready, skip onboarding
    const checkSetup = async () => {
      try {
        const res = await fetch(STATUS_API);
        if (res.ok) {
          const data = await res.json();
          if (data.state === "complete") {
            setSetupComplete(true);
            setChecking(false);
            return;
          }
        }
      } catch {
        // Backend might not be reachable — show app anyway
      }
      setChecking(false);
    };
    checkSetup();
  }, []);

  // Show loading spinner while checking
  if (checking) {
    return (
      <div style={{
        minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center',
        background: 'linear-gradient(135deg, #1B5E20, #2E7D32, #388E3C)',
      }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{
            width: 64, height: 64, borderRadius: '50%', backgroundColor: 'rgba(255,255,255,0.2)',
            display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 16px',
          }}>
            <span style={{ fontSize: 32 }}>🌱</span>
          </div>
          <div style={{ color: 'rgba(255,255,255,0.8)', fontSize: 14, fontWeight: 500 }}>
            Loading PotatoDoc...
          </div>
        </div>
      </div>
    );
  }

  // Show onboarding if setup is not complete
  if (!setupComplete) {
    return <OnboardingScreen onComplete={() => setSetupComplete(true)} />;
  }

  return <HomePage />;
}

export default App;
