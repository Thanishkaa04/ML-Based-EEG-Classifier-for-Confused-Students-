import React, { useMemo } from 'react';

const MODELS = [
  { id: 'cnn_rnn', name: 'CNN-RNN Hybrid' },
  { id: 'rf', name: 'Random Forest' },
  { id: 'enhanced_nn', name: 'Enhanced NN' },
  { id: 'rnn', name: 'RNN' },
  { id: 'resnet', name: 'ResNet' },
];

export default function ConfusionMatrixBoard({ streamHistory }) {
  const matrices = useMemo(() => {
    if (!streamHistory || streamHistory.length === 0) return null;
    
    // Initialize 2x2 matrix for each model (Binary Classification: 0 or 1)
    const mats = {};
    MODELS.forEach(m => {
      mats[m.id] = Array(2).fill(0).map(() => Array(2).fill(0));
    });
    
    // Populate counts
    streamHistory.forEach(point => {
      const actual = point.groundTruth; // 0 or 1
      MODELS.forEach(m => {
        if (point.inference && point.inference[m.id] !== undefined) {
          // Check if prediction is under "pred" or "prediction" depending on the model backend logic
          const result = point.inference[m.id];
          const predicted = result.pred !== undefined ? result.pred : result.prediction;
          
          if (actual >= 0 && actual <= 1 && predicted >= 0 && predicted <= 1) {
            mats[m.id][actual][predicted] += 1;
          }
        }
      });
    });
    
    return mats;
  }, [streamHistory]);

  if (!matrices) return null;

  return (
    <div className="glass-panel" style={{ marginTop: '1rem' }}>
      <h2 className="panel-title" style={{ marginBottom: '1rem' }}>Post-Stream Confusion Matrices (Binary Classification)</h2>
      <p className="text-muted" style={{ marginBottom: '1rem', fontSize: '0.9rem' }}>
        0 = Not Confused, 1 = Confused
      </p>
      <div style={{ display: 'flex', gap: '2rem', overflowX: 'auto', paddingBottom: '1rem' }}>
        {MODELS.map(model => {
          const mat = matrices[model.id];
          // Find max value for color scaling
          let maxVal = 0;
          mat.forEach(row => row.forEach(val => { if(val > maxVal) maxVal = val; }));
          
          return (
            <div key={model.id} style={{ minWidth: '150px' }}>
              <h3 style={{ fontSize: '1rem', marginBottom: '0.5rem', textAlign: 'center' }}>{model.name}</h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', border: '1px solid #333', padding: '8px', backgroundColor: '#1a1a1a', borderRadius: '8px' }}>
                <div style={{ display: 'flex', justifyContent: 'center', fontSize: '0.75rem', color: '#888', marginBottom: '4px' }}>Predicted &rarr;</div>
                <div style={{ display: 'flex', gap: '4px' }}>
                  <div style={{ width: '20px' }}></div>
                  <div style={{ width: '40px', textAlign: 'center', fontSize: '0.7rem', color: '#888' }}>0</div>
                  <div style={{ width: '40px', textAlign: 'center', fontSize: '0.7rem', color: '#888' }}>1</div>
                </div>
                {mat.map((row, actualIdx) => (
                  <div key={`row-${actualIdx}`} style={{ display: 'flex', gap: '4px' }}>
                    <div style={{ width: '20px', fontSize: '0.7rem', color: '#888', display: 'flex', alignItems: 'center', justifyContent: 'flex-end', paddingRight: '4px' }}>
                      {actualIdx}
                    </div>
                    {row.map((val, predIdx) => {
                      const intensity = maxVal > 0 ? val / maxVal : 0;
                      const isCorrect = actualIdx === predIdx;
                      const baseColor = isCorrect ? `rgba(34, 197, 94, ${intensity})` : `rgba(239, 68, 68, ${intensity})`;
                      const displayVal = val > 0 ? val : '0';
                      
                      return (
                        <div 
                          key={`cell-${actualIdx}-${predIdx}`}
                          style={{
                            width: '40px', 
                            height: '40px', 
                            backgroundColor: val > 0 ? baseColor : '#222',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            fontSize: '0.85rem',
                            color: '#fff',
                            borderRadius: '4px',
                            fontWeight: 'bold',
                            border: '1px solid rgba(255,255,255,0.1)'
                          }}
                          title={`Actual: ${actualIdx}, Predicted: ${predIdx} (Count: ${val})`}
                        >
                          {displayVal}
                        </div>
                      )
                    })}
                  </div>
                ))}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  );
}
