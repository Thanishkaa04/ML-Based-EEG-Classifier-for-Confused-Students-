import React, { useRef, useState } from 'react';
import { UploadCloud, Play } from 'lucide-react';

export default function UploadPanel({ onUploadSuccess }) {
  const fileInput = useRef(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleUploadClick = () => {
    fileInput.current.click();
  };

  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://localhost:8001/api/upload", {
        method: "POST",
        body: formData
      });
      const data = await res.json();
      
      if (!res.ok) {
        setError(data.detail || "Upload failed");
      } else {
        onUploadSuccess(data.avg_latency); // Optionally pass latency or results up
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
      if (fileInput.current) fileInput.current.value = "";
    }
  };

  const runSample = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("http://localhost:8001/api/sample");
      const data = await res.json();
      if (!res.ok) {
        setError(data.detail || "Sample run failed");
      } else {
        const firstRes = data.results[0];
        onUploadSuccess(data.results); // You can modify depending on how App handles it
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const [showSchema, setShowSchema] = useState(false);
  const [schemaData, setSchemaData] = useState(null);

  const fetchSchema = async () => {
    if (schemaData) {
      setShowSchema(!showSchema);
      return;
    }
    setLoading(true);
    try {
      const res = await fetch("http://localhost:8001/api/sample-schema");
      const data = await res.json();
      setSchemaData(data);
      setShowSchema(true);
    } catch (err) {
      setError("Failed to fetch sample schema");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="glass-panel" style={{ height: 'fit-content' }}>
      <h2 className="panel-title">Custom Data Pipeline</h2>
      <p className="text-muted" style={{ marginBottom: '1.5rem', lineHeight: '1.5' }}>
        AURA-BCI allows researchers to upload unseen EEG datasets to pass through the models for blind validation. 
        Please upload a CSV containing exactly 11 standardized features per row (Attention, Mediation, Raw, Delta, Theta, etc.).
      </p>

      <div className="drop-zone" onClick={handleUploadClick}>
        <UploadCloud size={48} className="drop-icon" />
        <h3>Click or Drop CSV File</h3>
        <p className="text-muted" style={{ marginTop: '0.5rem', fontSize: '0.875rem' }}>
          Must contain 11 features corresponding to a single-channel EEG.
        </p>
        <input  
          type="file" 
          accept=".csv" 
          ref={fileInput} 
          style={{ display: 'none' }} 
          onChange={handleFileChange} 
        />
      </div>

      {error && (
        <div className="error-msg">
          <strong>Validation Error:</strong> {error}
        </div>
      )}

      {loading && (
        <div style={{ marginTop: '1rem', color: 'var(--accent-blue)', fontWeight: 600, textAlign: 'center' }}>
          Processing Request...
        </div>
      )}

      <div style={{ marginTop: '2rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <button 
          className="btn" 
          onClick={(e) => { e.stopPropagation(); runSample(); }}
          disabled={loading}
        >
          <Play size={16} /> Run Holdout Sample
        </button>
        
        <button 
          className="btn" 
          style={{ backgroundColor: 'transparent', border: '1px solid var(--border)', color: 'var(--text-main)' }}
          onClick={(e) => { e.preventDefault(); fetchSchema(); }}
        >
          {showSchema ? "Hide Data Format" : "View Expected Data Format"}
        </button>
      </div>
      
      {showSchema && schemaData && (
        <div style={{ marginTop: '1.5rem', backgroundColor: '#111', padding: '1rem', borderRadius: '8px', border: '1px solid #333' }}>
          <h4 style={{ marginBottom: '0.5rem', color: 'var(--accent-purple)' }}>Expected 11-Feature Data Format:</h4>
          <p className="text-muted" style={{ fontSize: '0.85rem', marginBottom: '1rem' }}>
            Your CSV should have columns matching these exact keys, representing standardized power band outputs from a single-channel EEG sensor.
          </p>
          <pre style={{ maxHeight: '300px', overflowY: 'auto', fontSize: '0.8rem', color: '#a5b4fc', whiteSpace: 'pre-wrap' }}>
            {JSON.stringify(schemaData, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}
