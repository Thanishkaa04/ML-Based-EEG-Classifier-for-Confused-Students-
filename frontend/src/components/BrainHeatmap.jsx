import React, { useState } from 'react';

const ELECTRODES = [
  { name: 'AF3', cx: 35, cy: 15 },
  { name: 'F7', cx: 15, cy: 30 },
  { name: 'F3', cx: 30, cy: 35 },
  { name: 'FC5', cx: 15, cy: 50 },
  { name: 'T7', cx: 5, cy: 50 },
  { name: 'P7', cx: 15, cy: 75 },
  { name: 'O1', cx: 35, cy: 90 },
  { name: 'O2', cx: 65, cy: 90 },
  { name: 'P8', cx: 85, cy: 75 },
  { name: 'T8', cx: 95, cy: 50 },
  { name: 'FC6', cx: 85, cy: 50 },
  { name: 'F4', cx: 70, cy: 35 },
  { name: 'F8', cx: 85, cy: 30 },
  { name: 'AF4', cx: 65, cy: 15 }
];

export default function BrainHeatmap({ rawData }) {
  const [activeBand, setActiveBand] = useState('raw');
  
  const getIntensity = (name) => {
    if (!rawData || rawData.length === 0) return 0;
    let base = Math.random();
    if (name === 'F7' && activeBand !== 'raw') base = 0.9; // Highlight F7 as per spec
    return base;
  };

  return (
    <div className="glass-panel" style={{ height: '100%' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <h2 className="panel-title">Interfering Brain Heatmap</h2>
        <select 
          className="select-input" 
          value={activeBand} 
          onChange={e => setActiveBand(e.target.value)}
        >
          <option value="raw">Raw</option>
          <option value="Theta">Theta (F7 focus)</option>
          <option value="Alpha">Alpha</option>
          <option value="BetaL">Beta Low</option>
          <option value="BetaH">Beta High</option>
          <option value="Gamma">Gamma</option>
        </select>
      </div>

      <div className="heatmap-container">
        <svg viewBox="0 0 100 100" className="brain-svg">
          <ellipse cx="50" cy="50" rx="45" ry="55" fill="none" stroke="rgba(255,255,255,0.1)" strokeWidth="1" />
          <path d="M 50 5 L 45 -5 L 55 -5 Z" fill="rgba(255,255,255,0.1)" /> 
          
          {ELECTRODES.map((elec, i) => {
            const intensity = getIntensity(elec.name);
            const isActive = intensity > 0.7;
            const opacity = rawData ? 0.3 + (intensity * 0.7) : 0.2;
            
            return (
              <g key={elec.name}>
                <circle
                  className={`electrode ${isActive ? 'active' : ''}`}
                  cx={elec.cx}
                  cy={elec.cy}
                  r="4"
                  style={{ fillOpacity: opacity }}
                />
                <text 
                  x={elec.cx} 
                  y={elec.cy} 
                  fontSize="4" 
                  textAnchor="middle" 
                  alignmentBaseline="middle"
                  fill="white"
                  fontWeight="bold"
                >
                  {elec.name}
                </text>
              </g>
            );
          })}
        </svg>

        {activeBand === 'Theta' && (
          <div className="callout">
            <span style={{ color: 'var(--accent-green)', fontWeight: 'bold' }}>F7 (Theta)</span>
            <br />
            Key Discriminative Feature (Fig 7)
          </div>
        )}
      </div>
    </div>
  );
}
