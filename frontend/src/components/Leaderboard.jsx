import React from 'react';
import { Activity, Cpu, Zap, Database } from 'lucide-react';

const MODELS = [
  { id: 'cnn_rnn', name: 'CNN-RNN Hybrid', type: 'Spatiotemporal', accuracy: 65.70, latency: 45, icon: <Activity className="model-icon" style={{ color: '#a855f7' }} /> },
  { id: 'rf', name: 'Random Forest', type: 'Tree Ensemble', accuracy: 79.36, latency: 2, icon: <Cpu className="model-icon" style={{ color: '#6366f1' }} /> },
  { id: 'enhanced_nn', name: 'Enhanced Neural Net', type: 'Feedforward', accuracy: 69.22, latency: 15, icon: <Zap className="model-icon" style={{ color: '#ec4899' }} /> },
  { id: 'rnn', name: 'Recurrent Neural Net', type: 'Sequential', accuracy: 68.59, latency: 35, icon: <Database className="model-icon" style={{ color: '#3b82f6' }} /> },
  { id: 'resnet', name: 'Residual Network', type: 'Deep Spatial', accuracy: 66.13, latency: 60, icon: <Cpu className="model-icon" style={{ color: '#14b8a6' }} /> },
];

export default function Leaderboard({ inferenceData, groundTruth }) {
  if (!inferenceData) {
    return (
      <div className="glass-panel" style={{ marginTop: '2rem' }}>
        <h2 className="panel-title">Model Performance Leaderboard</h2>
        <p className="text-muted" style={{ marginBottom: '1.5rem', lineHeight: '1.5' }}>
          These models were trained on the 11-feature NeuroSky dataset to predict binary cognitive states.
        </p>
        <div className="leaderboard">
          <p className="text-muted">Waiting for data stream...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="glass-panel" style={{ marginTop: '2rem' }}>
      <h2 className="panel-title">Model Performance Leaderboard</h2>
      <p className="text-muted" style={{ marginBottom: '1.5rem', lineHeight: '1.5' }}>
        These models were trained on the 11-feature NeuroSky dataset to predict binary cognitive states.
      </p>
      <div className="leaderboard">
        {MODELS.map(model => {
          const result = inferenceData[model.id];
          if (!result) return null;
          
          const isCorrect = result.pred === groundTruth;
          const prob = (result.probs && result.probs[result.pred]) ? result.probs[result.pred] * 100 : 100;
          
          return (
            <div key={model.id} className="model-card">
              <div className="model-name">
                {model.name}
                <span className="sub">Val Acc: {model.accuracy}%</span>
              </div>
              <div className="model-pred">
                ID: {result.pred}
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', width: '100%' }}>
                <span style={{ fontSize: '0.7rem', color: '#888', marginBottom: '2px' }}>Confidence: {prob.toFixed(1)}%</span>
                <div className="prob-bar-container" title={`Confidence: ${prob.toFixed(1)}%`}>
                  <div className="prob-bar" style={{ width: `${prob}%` }}></div>
                </div>
              </div>
              <div className="model-latency">
                ~{model.latency}ms
              </div>
              <div className={`match-status ${isCorrect ? 'correct' : 'incorrect'}`}>
                {isCorrect ? 'MATCH' : 'MISS'}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
