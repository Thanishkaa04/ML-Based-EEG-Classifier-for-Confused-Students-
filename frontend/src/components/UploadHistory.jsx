import React, { useEffect, useState } from 'react';
import { Trash2 } from 'lucide-react';

export default function UploadHistory({ refreshTrigger }) {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchHistory();
  }, [refreshTrigger]);

  const fetchHistory = async () => {
    try {
      const res = await fetch('http://localhost:8001/api/history');
      if (res.ok) {
        const data = await res.json();
        setHistory(data);
      }
    } catch (e) {
      console.error("Failed to load history", e);
    } finally {
      setLoading(false);
    }
  };

  const deleteHistory = async (id) => {
    try {
      await fetch(`http://localhost:8000/api/history/${id}`, { method: 'DELETE' });
      fetchHistory();
    } catch (e) {
      console.error("Failed to delete history", e);
    }
  };

  if (loading) {
    return (
      <div className="glass-panel">
        <p>Loading history...</p>
      </div>
    );
  }

  return (
    <div className="glass-panel">
      <h2 className="panel-title">Upload History Logs</h2>
      {history.length === 0 ? (
        <p className="text-muted">No uploads recorded in database.</p>
      ) : (
        <div style={{ overflowX: 'auto' }}>
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>File Name</th>
                <th>Date</th>
                <th>Rows</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {history.map(log => (
                <tr key={log.id}>
                  <td>#{log.id}</td>
                  <td style={{ fontWeight: 600 }}>{log.filename}</td>
                  <td>{new Date(log.upload_time).toLocaleString()}</td>
                  <td>{log.num_rows}</td>
                  <td>
                    <button 
                      onClick={() => deleteHistory(log.id)}
                      style={{ background: 'transparent', border: 'none', color: '#f87171', cursor: 'pointer' }}
                    >
                      <Trash2 size={18} />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
