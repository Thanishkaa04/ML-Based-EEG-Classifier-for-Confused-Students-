import React, { useState, useEffect, useRef } from 'react';
import Leaderboard from './components/Leaderboard';
import BrainHeatmap from './components/BrainHeatmap';
import UploadPanel from './components/UploadPanel';
import UploadHistory from './components/UploadHistory';
import ConfusionMatrixBoard from './components/ConfusionMatrixBoard';
import { Play, Square, RotateCcw, Activity } from 'lucide-react';
import './index.css';

function App() {
  const [activeTab, setActiveTab] = useState('live');
  const [ws, setWs] = useState(null);
  
  // Live Replay State
  const [subjectId, setSubjectId] = useState(0);
  const [videoId, setVideoId] = useState(0);
  const [speedMs, setSpeedMs] = useState(100);
  const [isPlaying, setIsPlaying] = useState(false);
  const [inferenceData, setInferenceData] = useState(null);
  const [groundTruth, setGroundTruth] = useState(null);
  const [rawData, setRawData] = useState(null);
  const [progress, setProgress] = useState({ current: 0, total: 0 });
  
  // Upload State
  const [refreshHistory, setRefreshHistory] = useState(0);

  // New States
  const [validSessions, setValidSessions] = useState({});
  const [streamHistory, setStreamHistory] = useState([]);

  useEffect(() => {
    fetch('http://localhost:8001/api/valid-sessions')
      .then(res => res.json())
      .then(data => {
        setValidSessions(data);
        if (data['0'] && data['0'].length > 0) {
           setVideoId(data['0'][0]);
        }
      })
      .catch(err => console.error("Failed to load valid sessions:", err));
      
    return () => {
      if (ws) {
        ws.close();
      }
    };
  }, [ws]);

  const startStream = () => {
    if (ws) ws.close();
    setStreamHistory([]); // reset history
    
    const socket = new WebSocket('ws://localhost:8001/ws/eeg');
    
    socket.onopen = () => {
      setIsPlaying(true);
      socket.send(JSON.stringify({
        subject_id: parseInt(subjectId),
        video_id: parseInt(videoId),
        speed_ms: speedMs
      }));
    };
    
    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.status === 'done' || data.error) {
        setIsPlaying(false);
        socket.close();
        if (data.error) alert(data.error);
        return;
      }
      
      setRawData(data.raw_data);
      setInferenceData(data.inference);
      setGroundTruth(data.ground_truth);
      if (data.current_row && data.total_rows) {
        setProgress({ current: data.current_row, total: data.total_rows });
      }
      
      // Track history
      setStreamHistory(prev => [...prev, { inference: data.inference, groundTruth: data.ground_truth }]);
    };
    
    socket.onclose = () => {
      setIsPlaying(false);
    };
    
    setWs(socket);
  };

  const stopStream = () => {
    if (ws) {
      ws.close();
      setWs(null);
    }
    setIsPlaying(false);
  };

  const resetStream = () => {
    stopStream();
    setInferenceData(null);
    setRawData(null);
    setGroundTruth(null);
    setProgress({ current: 0, total: 0 });
  };

  return (
    <div className="app-container">
      <header className="header" style={{ justifyContent: 'center' }}>
        <div className="tabs" style={{ width: '100%', justifyContent: 'center' }}>
          <button 
            className={`tab-btn ${activeTab === 'live' ? 'active' : ''}`}
            onClick={() => setActiveTab('live')}
          >
            Live Replay
          </button>
          <button 
            className={`tab-btn ${activeTab === 'upload' ? 'active' : ''}`}
            onClick={() => setActiveTab('upload')}
          >
            Upload & Inspect
          </button>
        </div>
      </header>

      <main className="main-content">
        {activeTab === 'live' ? (
          <>
            <div className="glass-panel" style={{ marginBottom: '1rem' }}>
              <h2 className="panel-title">Real-Time BCI Deployment Demo (Binary)</h2>
              <p className="text-muted" style={{ lineHeight: '1.5' }}>
                This dashboard demonstrates the live deployment of 5 machine learning models on a single-sensor dataset. 
                Select a Subject and a Video Topic they watched. The backend streams 11 real-time features (Alpha, Beta, Gamma, etc.) 
                over a WebSocket, and all 5 neural architectures run real-time binary inference (predicting 0: Not Confused, 1: Confused).
              </p>
            </div>
            
            <div className="live-dashboard-grid">
              {/* Top Controls */}
              <div className="glass-panel controls-panel">
              <div className="control-group">
                <div>
                  <label style={{ marginRight: '0.5rem', fontSize: '0.875rem' }}>Subject:</label>
                  <select 
                    className="select-input" 
                    value={subjectId} 
                    onChange={e => {
                      const newSub = e.target.value;
                      setSubjectId(newSub);
                      if (validSessions[newSub] && validSessions[newSub].length > 0) {
                        setVideoId(validSessions[newSub][0]);
                      }
                    }} 
                    disabled={isPlaying}
                  >
                    {Object.keys(validSessions).map(n => <option key={n} value={n}>Subject {n}</option>)}
                  </select>
                </div>
                <div>
                  <label style={{ marginRight: '0.5rem', fontSize: '0.875rem' }}>Video:</label>
                  <select className="select-input" value={videoId} onChange={e => setVideoId(e.target.value)} disabled={isPlaying}>
                    {(validSessions[subjectId] || []).map(n => <option key={n} value={n}>Topic {n}</option>)}
                  </select>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginLeft: '1rem' }}>
                  <label style={{ fontSize: '0.875rem' }}>Speed: {speedMs}ms</label>
                  <input 
                    type="range" 
                    min="50" max="500" step="50" 
                    value={speedMs} 
                    onChange={e => setSpeedMs(Number(e.target.value))}
                    disabled={isPlaying}
                  />
                </div>
              </div>
              <div className="control-group">
                {!isPlaying ? (
                  <button className="btn btn-primary" onClick={startStream}>
                    <Play size={18} /> Stream Replay
                  </button>
                ) : (
                  <button className="btn" onClick={stopStream} style={{ borderColor: '#f87171', color: '#f87171' }}>
                    <Square size={18} /> Stop
                  </button>
                )}
                <button className="btn" onClick={resetStream}>
                  <RotateCcw size={18} /> Reset
                </button>
              </div>
            </div>

            {/* Leaderboard */}
            <div style={{ gridColumn: '1 / 3' }}>
              {progress.total > 0 && (
                <div style={{ marginBottom: '1.5rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.875rem', color: '#a5b4fc', marginBottom: '4px' }}>
                    <span>Stream Progress</span>
                    <span>{progress.current} / {progress.total} timesteps</span>
                  </div>
                  <div style={{ width: '100%', height: '8px', backgroundColor: '#333', borderRadius: '4px', overflow: 'hidden' }}>
                    <div style={{ height: '100%', backgroundColor: '#6366f1', width: `${(progress.current / progress.total) * 100}%`, transition: 'width 0.1s linear' }}></div>
                  </div>
                </div>
              )}
              <Leaderboard inferenceData={inferenceData} groundTruth={groundTruth} />
            </div>
          </div>
          
          {streamHistory.length > 0 && !isPlaying && (
            <ConfusionMatrixBoard streamHistory={streamHistory} />
          )}
          </>
        ) : (
          <div className="upload-grid">
            <UploadPanel onUploadSuccess={() => setRefreshHistory(prev => prev + 1)} />
            <UploadHistory refreshTrigger={refreshHistory} />
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
