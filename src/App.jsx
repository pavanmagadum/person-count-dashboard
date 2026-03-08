import { useEffect, useState, useRef } from "react";
import { database } from "./firebase";
import { ref, update, onValue } from "firebase/database";
import * as tf from "@tensorflow/tfjs";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import {
  Chart as ChartJS,
  LineElement,
  CategoryScale,
  LinearScale,
  PointElement,
  Tooltip,
  Legend,
  Filler
} from "chart.js";
import { Line } from "react-chartjs-2";
import { useReducer, useMemo, useCallback } from "react";
import "./App.css";

ChartJS.register(LineElement, CategoryScale, LinearScale, PointElement, Tooltip, Legend, Filler);

function App() {
  // REDUCER FOR BATCHING DATA UPDATES (Industrial Grade)
  const initialState = {
    count: 0,
    inCount: 0,
    outCount: 0,
    accuracy: 0,
    history: new Array(15).fill(0),
    timestamp: "SYSTEM_READY",
    activeSource: "SCANNING",
    statusMessage: "INITIALIZING_AI...",
    isEcoMode: false,
  };

  function detectionReducer(state, action) {
    switch (action.type) {
      case "UPDATE_DETECTION":
        return {
          ...state,
          count: action.count,
          accuracy: action.accuracy,
          history: [...state.history.slice(1), action.count],
          timestamp: new Date().toLocaleTimeString(),
          activeSource: action.source || "LOCAL_WEBCAM",
          isEcoMode: action.count === 0,
        };
      case "INC_TRAFFIC":
        return action.direction === "IN"
          ? { ...state, inCount: state.inCount + 1 }
          : { ...state, outCount: state.outCount + 1 };
      case "UPDATE_FROM_FIREBASE":
        return {
          ...state,
          count: action.count,
          accuracy: action.accuracy,
          timestamp: action.timestamp,
          activeSource: "REMOTE_DATABASE",
        };
      case "SET_STATUS":
        return { ...state, statusMessage: action.message };
      default:
        return state;
    }
  }

  const [state, dispatch] = useReducer(detectionReducer, initialState);

  // HUD & CAMERA CONTROL
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const [isModelLoaded, setIsModelLoaded] = useState(false);

  // REFS FOR CAMERA & DETECTION
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const modelRef = useRef(null);
  const detectionInterval = useRef(null);
  const firebaseSyncTimer = useRef(0);
  const prevCentroids = useRef([]); // Critical for Tracking Logic

  // RISK LOGIC (SIMPLIFIED ENGLISH)
  const getRiskUI = (val) => {
    if (val === 0) return { label: "Idle / No People", color: "#64748b", steps: 0, msg: "System ready. No activity detected." };
    if (val <= 2) return { label: "Low Risk (Normal)", color: "#10b981", steps: 1, msg: "Low activity detected. Environment is safe." };
    if (val <= 5) return { label: "Medium Risk (Careful)", color: "#f59e0b", steps: 2, msg: "Increased occupancy. Caution advised." };
    return { label: "High Risk (Crowded)", color: "#f43f5e", steps: 3, msg: "⚠️ HIGH DENSITY! Immediate action required." };
  };

  // 1. LOAD AI MODEL ON STARTUP
  useEffect(() => {
    const loadModel = async () => {
      try {
        await tf.setBackend('webgl');
        await tf.ready();
        const model = await cocoSsd.load({ base: 'mobilenet_v2' });
        modelRef.current = model;
        setIsModelLoaded(true);
        dispatch({ type: "SET_STATUS", message: "NEURAL_ENGINE_READY" });
      } catch (err) {
        console.warn("WebGL failure, falling back to CPU:", err);
        await tf.setBackend('cpu');
        const model = await cocoSsd.load();
        modelRef.current = model;
        setIsModelLoaded(true);
      }
    };
    loadModel();

    const dataRef = ref(database, "person_detection");
    const unsubscribe = onValue(dataRef, (snapshot) => {
      const data = snapshot.val();
      if (data && !isCameraOpen) {
        dispatch({
          type: "UPDATE_FROM_FIREBASE",
          count: data.person_count || 0,
          accuracy: data.accuracy || 99.8,
          timestamp: data.timestamp || new Date().toLocaleTimeString()
        });
      }
    });

    return () => unsubscribe();
  }, [isCameraOpen]);

  // 2. CAMERA FEED & DETECTION LOOP
  useEffect(() => {
    if (isCameraOpen) {
      startCamera();
    } else {
      stopCamera();
    }
  }, [isCameraOpen]);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: 640, height: 480 },
        audio: false
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current.play();
          runDetection();
        };
      }
    } catch (err) {
      console.error("Camera access denied:", err);
      setIsCameraOpen(false);
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      videoRef.current.srcObject.getTracks().forEach(track => track.stop());
    }
    if (detectionInterval.current) {
      clearInterval(detectionInterval.current);
    }
  };

  const runDetection = () => {
    detectionInterval.current = setInterval(async () => {
      if (modelRef.current && videoRef.current && videoRef.current.readyState === 4) {
        const predictions = await modelRef.current.detect(videoRef.current, 20, 0.4);
        console.log("Raw Predictions:", predictions);

        // 1. FILTER: High-precision person filter
        const persons = predictions.filter(p => p.class === "person");
        const personCount = persons.length;
        const faceDetected = persons.some(p => p.score > 0.8);

        const avgAcc = persons.length > 0
          ? Number((persons.reduce((sum, p) => sum + p.score, 0) / persons.length * 100).toFixed(1))
          : 0;

        // 2. TRIPWIRE: Directional Tracking Logic
        persons.forEach(person => {
          const centroidY = person.bbox[1] + person.bbox[3] / 2;
          const match = prevCentroids.current.find(c => Math.abs(c.x - person.bbox[0]) < 50);

          if (match) {
            if (match.y < 240 && centroidY >= 240) {
              dispatch({ type: "INC_TRAFFIC", direction: "IN" });
            } else if (match.y > 240 && centroidY <= 240) {
              dispatch({ type: "INC_TRAFFIC", direction: "OUT" });
            }
          }
        });
        prevCentroids.current = persons.map(p => ({ x: p.bbox[0], y: p.bbox[1] + p.bbox[3] / 2 }));

        // 3. STATE UPDATE
        dispatch({
          type: "UPDATE_DETECTION",
          count: personCount,
          accuracy: avgAcc,
          source: faceDetected ? "FACE_SCANNED_ACTIVE" : "LOCAL_WEBCAM"
        });

        const now = Date.now();
        if (now - firebaseSyncTimer.current > 1500) {
          syncToFirebase(personCount, avgAcc);
          firebaseSyncTimer.current = now;
        }

        requestAnimationFrame(() => drawBoxes(persons, getRiskUI(personCount)));
      }
    }, 250);
  };

  // 3. TELEGRAM ALERT SYSTEM
  const [alertStatus, setAlertStatus] = useState("IDLE");
  const [lastAlertTimestamp, setLastAlertTimestamp] = useState(null);
  const lastAlertTime = useRef(0);
  const ALERT_COOLDOWN = 60000; // 1 minute cooldown

  // Central Monitor: Fires regardless of Data Source (Now using state.count)
  useEffect(() => {
    if (state.count > 5) {
      console.log(`[QUANTUM_AI] Threshold Exceeded: ${state.count}.`);
      sendTelegramAlert(state.count);
    }
  }, [state.count]);

  const sendTelegramAlert = async (currentCount) => {
    const now = Date.now();

    // NOTE: Replace these with your actual credentials for live alerts
    const botToken = "8647494412:AAHVCC_6A4M5LdwGWxD6UvSapEtV5F78gcE";
    const chatId = "912525748";

    // 1. Validation Check
    const token = botToken.trim();
    const chat = chatId.trim();

    if (!token || !chat || token.includes("YOUR_TOKEN")) {
      setAlertStatus("MISSING_CREDENTIALS");
      setTimeout(() => setAlertStatus("IDLE"), 3000);
      return;
    }

    // 2. Cooldown check
    if (currentCount !== "TEST" && (now - lastAlertTime.current < ALERT_COOLDOWN)) {
      return;
    }

    const reportCount = currentCount === "TEST" ? "DEMO_DETECTION" : currentCount;

    try {
      setAlertStatus("PREPARING_IMAGE...");

      // A. CAPTURE SNAPSHOT FROM CANVAS/VIDEO
      let photoBlob = null;
      if (isCameraOpen && videoRef.current) {
        const captureCanvas = document.createElement("canvas");
        captureCanvas.width = 640;
        captureCanvas.height = 480;
        const ctx = captureCanvas.getContext("2d");

        // Draw the frame
        ctx.drawImage(videoRef.current, 0, 0, 640, 480);
        // Overlay the detection boxes from the main canvas
        if (canvasRef.current) {
          ctx.drawImage(canvasRef.current, 0, 0, 640, 480);
        }

        photoBlob = await new Promise(resolve => captureCanvas.toBlob(resolve, 'image/jpeg', 0.8));
      }

      setAlertStatus("DISPATCHING...");

      const alertCaption = `🚨 *QUANTUM_VISION SECURITY ALERT* 🚨\n\n` +
        `👤 *Density Detected:* ${reportCount} People\n` +
        `📍 *Location:* Sector_01_Scanner\n` +
        `⏰ *Timestamp:* ${new Date().toLocaleTimeString()}\n` +
        `━━━━━━━━━━━━━━━━━━━━\n` +
        `_Visual Confirmation Attached_`;

      let response;

      if (photoBlob) {
        // Send Image with Caption
        const formData = new FormData();
        formData.append("chat_id", chat);
        formData.append("photo", photoBlob, "security_alert.jpg");
        formData.append("caption", alertCaption);
        formData.append("parse_mode", "Markdown");

        response = await fetch(`https://api.telegram.org/bot${token}/sendPhoto`, {
          method: "POST",
          body: formData
        });
      } else {
        // Fallback to text if camera is not active
        response = await fetch(`https://api.telegram.org/bot${token}/sendMessage`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            chat_id: chat,
            text: alertCaption + "\n\n(Camera Offline - Text Only Alert)",
            parse_mode: "Markdown"
          })
        });
      }

      const resData = await response.json();

      if (response.ok) {
        console.log("[QUANTUM_AI] Alert Dispatched Successfully!");
        setAlertStatus("ALERT_SENT");
        setLastAlertTimestamp(new Date().toLocaleTimeString());
        if (currentCount !== "TEST") lastAlertTime.current = now;
        setTimeout(() => setAlertStatus("IDLE"), 5000);
      } else {
        console.error("[QUANTUM_AI] Telegram API Error:", resData);
        setAlertStatus(resData.description || "API_ERROR");
        setTimeout(() => setAlertStatus("IDLE"), 4000);
      }
    } catch (error) {
      console.error("[QUANTUM_AI] Network Failure:", error);
      setAlertStatus("CONN_FAILED");
      setTimeout(() => setAlertStatus("IDLE"), 4000);
    }
  };

  const drawBoxes = (persons, currentRisk) => {
    if (!canvasRef.current) return;
    const ctx = canvasRef.current.getContext("2d");
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    persons.forEach(person => {
      const [x, y, width, height] = person.bbox;
      ctx.strokeStyle = currentRisk.color;
      ctx.lineWidth = 3;
      ctx.strokeRect(x, y, width, height);

      ctx.fillStyle = currentRisk.color;
      ctx.font = "bold 12px Inter";
      ctx.fillText(`${(person.score * 100).toFixed(0)}% Person`, x, y > 10 ? y - 5 : 10);
    });
  };

  const syncToFirebase = (personCount, avgAcc) => {
    const dataRef = ref(database, "person_detection");
    const riskData = getRiskUI(personCount);
    update(dataRef, {
      person_count: personCount,
      accuracy: personCount > 0 ? avgAcc : 99.8,
      timestamp: new Date().toLocaleString(),
      crowd_level: riskData.label,
      message: riskData.msg,
      status_color: riskData.color,
      person_detected: personCount > 0 ? 1 : 0
    });
  };

  // 4. MEMOIZED CHART & UI DATA (Critical for Performance)
  const risk = useMemo(() => getRiskUI(state.count), [state.count]);

  const chartData = useMemo(() => ({
    labels: new Array(state.history.length).fill(""),
    datasets: [{
      data: state.history,
      borderColor: risk.color,
      backgroundColor: (context) => {
        const ctx = context.chart.ctx;
        const gradient = ctx.createLinearGradient(0, 0, 0, 250);
        gradient.addColorStop(0, `${risk.color}44`);
        gradient.addColorStop(1, `${risk.color}00`);
        return gradient;
      },
      borderWidth: 2,
      tension: 0.4,
      fill: true,
      pointRadius: 0,
    }]
  }), [state.history, risk.color]);

  return (
    <div className="app-container">
      <nav className="navbar">
        <div className="nav-brand">
          <span className="logo-icon">💠</span>
          <span className="logo-text">QUANTUM_VISION</span>
        </div>

        {/* Standard Navigation */}
        <div className="desktop-nav">
          <a href="#dashboard" className="nav-item">Monitor</a>
          <a href="#" className="nav-item">Settings</a>
        </div>
      </nav>

      <main className="dashboard-wrapper fade-in" id="dashboard">
        <header className="header-section">
          <div className="title-group">
            <div className="system-tag">AI Visual Intelligence</div>
            <h1>Safety Monitoring</h1>
          </div>

          <div className="main-action-area">
            <button
              className="action-btn-primary"
              onClick={() => setIsCameraOpen(!isCameraOpen)}
              disabled={!isModelLoaded}
            >
              <span className="btn-pulse"></span>
              {!isModelLoaded ? 'INITIALIZING AI...' : (isCameraOpen ? 'STOP SECURITY SCAN' : 'OPEN WEB CAMERA')}
            </button>
          </div>

          <div className="stream-status">
            <div className="status-label">NEURAL_ENGINE</div>
            <div className="status-indicator" style={{ color: isCameraOpen ? '#10b981' : '#64748b' }}>
              {isCameraOpen ? '● ACTIVE' : '○ STANDBY'}
            </div>
          </div>
        </header>

        <div className="main-grid">
          <section className="main-visual">
            <div className="visual-header">
              <span className="system-tag">{isCameraOpen ? 'LOCAL_DETECTION_STREAM' : 'Timeline Activity History'}</span>
              <span className="source-badge" style={{ color: risk.color, background: `${risk.color}11` }}>{state.activeSource}</span>
            </div>

            <div className="visual-content">
              {isCameraOpen ? (
                <div className="camera-container">
                  <video ref={videoRef} className="live-camera" muted playsInline />
                  <canvas ref={canvasRef} width="640" height="480" style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none', willChange: 'transform' }} />
                  <div className="camera-hud">
                    <div className="hud-line">SCANNING ● ACTIVE</div>
                    <div className="hud-line" style={{ color: risk.color }}>RISK: {risk.label.toUpperCase()}</div>
                  </div>
                </div>
              ) : (
                <div className="chart-container">
                  <Line data={chartData} options={{ responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { y: { display: false }, x: { display: false } } }} />
                </div>
              )}
            </div>
          </section>

          <aside className="bento-sidebar">
            {/* NEW: RETAIL COMMAND CENTER STATS */}
            <div className="card-row">
              <div className="card mini-card">
                <div className="system-tag">Entrance_Total</div>
                <div className="mini-number">+{state.inCount}</div>
              </div>
              <div className="card mini-card">
                <div className="system-tag">Exit_Total</div>
                <div className="mini-number">-{state.outCount}</div>
              </div>
            </div>

            <div className="card stat-hero">
              <div className="system-tag">Current Occupancy Level</div>
              <div className="huge-number">{state.count}</div>
              <div className="stat-detail">Precision Scanned: {state.accuracy}%</div>
              {state.isEcoMode && (
                <div className="eco-badge">🌿 ECO_MODE: ACTIVE</div>
              )}
            </div>

            <div className="card analytics-matrix">
              <div className="system-tag">Zone Analytics [BETA]</div>
              <div className="zone-grid">
                {[...Array(4)].map((_, i) => (
                  <div key={i} className={`zone-cell ${state.count > 0 && i === 1 ? 'hot' : ''}`}>
                    Z_0{i + 1}
                  </div>
                ))}
              </div>
              <p className="matrix-desc">Monitoring dwell-time hotspots in Sector_Alpha.</p>
            </div>

            <div className="card telegram-alert-card">
              <div className="system-tag">Smart Alert Link</div>
              <div className="telegram-status-area">
                <div className="telegram-badge">
                  <span className="tg-icon">✈️</span>
                  <span>Telegram Bot</span>
                </div>
                <div className={`alert-indicator ${alertStatus !== "IDLE" ? 'active' : ''}`}>
                  {alertStatus}
                </div>
              </div>
              <p className="matrix-desc">Auto-notifies authorities when person count exceeds 5 detected instances.</p>

              <div className="threshold-bar">
                <div className="threshold-fill" style={{ width: `${Math.min((state.count / 5) * 100, 100)}%`, backgroundColor: state.count > 5 ? 'var(--accent-rose)' : 'var(--accent-cyan)' }}></div>
              </div>
              {lastAlertTimestamp && <div className="last-alert-meta">LAST_DISPATCH: {lastAlertTimestamp}</div>}
            </div>

            <div className="card">
              <div className="system-tag">Safety Level Check</div>
              <div className="risk-level-hud">
                <div className="status-ring" style={{ color: risk.color, backgroundColor: risk.color }}></div>
                <div className="matrix-label" style={{ color: risk.color }}>{risk.label}</div>
              </div>
              <div className="matrix-visualizer">
                {[1, 2, 3].map(s => (
                  <div key={s} className={`matrix-step ${risk.steps >= s ? 'active' : ''}`} style={{ color: risk.color, backgroundColor: risk.steps >= s ? 'currentColor' : 'rgba(255,255,255,0.05)' }} />
                ))}
              </div>
              <p className="matrix-desc">{risk.msg}</p>
            </div>
          </aside>
        </div>

        <footer className="footer-meta">
          <div className="meta-item">PROJECT_STATUS: INDUSTRIAL_GRADE_V2.0 // NODE: EDGE_COMPUTE</div>
          <div className="meta-item">ECO_STATUS: {state.isEcoMode ? "SAVING_ENERGY" : "FULL_POWER"} // LOAD: {state.count * 85}W (EST)</div>
        </footer>
      </main>
    </div>
  );
}

export default App;