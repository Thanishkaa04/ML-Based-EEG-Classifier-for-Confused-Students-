from fastapi import FastAPI, WebSocket, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import pandas as pd
import numpy as np
import torch
import joblib
import time
import json
import asyncio
from datetime import datetime

from models import EnhancedNN, RNNModel, ResNet, CNN_RNN

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

models = {}
scaler = None
df_full = None
FEATURE_COLS = ['Attention', 'Mediation', 'Raw', 'Delta', 'Theta', 'Alpha1', 'Alpha2', 'Beta1', 'Beta2', 'Gamma1', 'Gamma2']

def init_db():
    conn = sqlite3.connect("history.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS streams
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, subject_id TEXT, video_id TEXT, 
                  model_name TEXT, predictions TEXT, ground_truth INTEGER,
                  latency_ms REAL, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

@app.on_event("startup")
def startup_event():
    global models, scaler, df_full, FEATURE_COLS
    init_db()
    
    print("Loading SQLite DB...")
    
    print("Loading models...")
    try:
        scaler = joblib.load("models/scaler.joblib")
        models['rf'] = joblib.load("models/rf_model.joblib")
        
        models['enhanced_nn'] = EnhancedNN(num_classes=2, input_feature_size=36)
        models['enhanced_nn'].load_state_dict(torch.load("models/EnhancedNN.pth", map_location=torch.device('cpu'), weights_only=True))
        models['enhanced_nn'].eval()
        
        models['rnn'] = RNNModel(input_size=36, hidden_size=128, output_size=2)
        models['rnn'].load_state_dict(torch.load("models/RNNModel.pth", map_location=torch.device('cpu'), weights_only=True))
        models['rnn'].eval()
        
        models['resnet'] = ResNet(num_classes=2)
        models['resnet'].load_state_dict(torch.load("models/ResNet.pth", map_location=torch.device('cpu'), weights_only=True))
        models['resnet'].eval()
        
        models['cnn_rnn'] = CNN_RNN(num_classes=2)
        models['cnn_rnn'].load_state_dict(torch.load("models/CNN_RNN.pth", map_location=torch.device('cpu'), weights_only=True))
        models['cnn_rnn'].eval()
        print("All models loaded successfully.")
    except Exception as e:
        print("Warning: Could not load some models.", e)
        
    print("Loading dataset into memory for WebSocket replays...")
    try:
        df_full = pd.read_csv("../EEG_data.csv")
        base_cols = ['Attention', 'Mediation', 'Raw', 'Delta', 'Theta', 'Alpha1', 'Alpha2', 'Beta1', 'Beta2', 'Gamma1', 'Gamma2']
        epsilon = 1e-8
        df_full['Theta_Beta_Ratio'] = df_full['Theta'] / (df_full['Beta1'] + df_full['Beta2'] + epsilon)
        df_full['Theta_Alpha_Ratio'] = df_full['Theta'] / (df_full['Alpha1'] + df_full['Alpha2'] + epsilon)
        df_full['Delta_Theta_Ratio'] = df_full['Delta'] / (df_full['Theta'] + epsilon)

        def add_rolling_features(group):
            window = 5
            for col in base_cols:
                group[f'{col}_mean_5'] = group[col].rolling(window=window, min_periods=1).mean()
                group[f'{col}_std_5'] = group[col].rolling(window=window, min_periods=1).std().fillna(0)
            return group

        df_full = df_full.groupby(['SubjectID', 'VideoID']).apply(add_rolling_features).reset_index(drop=True)
        
        FEATURE_COLS = base_cols + ['Theta_Beta_Ratio', 'Theta_Alpha_Ratio', 'Delta_Theta_Ratio'] + [f"{col}_mean_5" for col in base_cols] + [f"{col}_std_5" for col in base_cols]
        print("Dataset loaded with 36 Engineered Features.")
    except Exception as e:
        print("Warning: Dataset could not be loaded:", e)

def run_inference(row_features):
    scaled = scaler.transform([row_features])
    tensor_input = torch.tensor(scaled, dtype=torch.float32)
    
    results = {}
    
    if 'rf' in models:
        start = time.perf_counter()
        rf_pred = models['rf'].predict(scaled)[0]
        rf_probs = models['rf'].predict_proba(scaled)[0].tolist()
        end = time.perf_counter()
        results['rf'] = {
            "pred": int(rf_pred),
            "latency_ms": (end - start) * 1000,
            "probs": rf_probs
        }
    
    with torch.no_grad():
        t0 = time.perf_counter()
        out_enn = models['enhanced_nn'](tensor_input)
        prob_enn = torch.softmax(out_enn, dim=1)[0].tolist()
        pred_enn = torch.argmax(out_enn, dim=1).item()
        t1 = time.perf_counter()
        results['enhanced_nn'] = {"pred": pred_enn, "latency_ms": (t1-t0)*1000, "probs": prob_enn}
        
        t0 = time.perf_counter()
        out_rnn = models['rnn'](tensor_input)
        prob_rnn = torch.softmax(out_rnn, dim=1)[0].tolist()
        pred_rnn = torch.argmax(out_rnn, dim=1).item()
        t1 = time.perf_counter()
        results['rnn'] = {"pred": pred_rnn, "latency_ms": (t1-t0)*1000, "probs": prob_rnn}
        
        t0 = time.perf_counter()
        out_resnet = models['resnet'](tensor_input)
        prob_resnet = torch.softmax(out_resnet, dim=1)[0].tolist()
        pred_resnet = torch.argmax(out_resnet, dim=1).item()
        t1 = time.perf_counter()
        results['resnet'] = {"pred": pred_resnet, "latency_ms": (t1-t0)*1000, "probs": prob_resnet}
        
        t0 = time.perf_counter()
        out_cnnrnn = models['cnn_rnn'](tensor_input)
        prob_cnnrnn = torch.softmax(out_cnnrnn, dim=1)[0].tolist()
        pred_cnnrnn = torch.argmax(out_cnnrnn, dim=1).item()
        t1 = time.perf_counter()
        results['cnn_rnn'] = {"pred": pred_cnnrnn, "latency_ms": (t1-t0)*1000, "probs": prob_cnnrnn}
        
    return results

@app.websocket("/ws/eeg")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        data = await websocket.receive_text()
        config = json.loads(data)
        subject_id = config.get("subject_id", 0)
        video_id = config.get("video_id", 0)
        speed_ms = config.get("speed_ms", 100)
        
        if df_full is None:
            await websocket.send_text(json.dumps({"error": "Dataset not loaded"}))
            return
            
        session_df = df_full[(df_full['SubjectID'] == subject_id) & (df_full['VideoID'] == video_id)]
        if session_df.empty:
            await websocket.send_text(json.dumps({"error": "No data found for subject/video"}))
            return
            
        total_rows = len(session_df)
        current_row = 0
        for _, row in session_df.iterrows():
            current_row += 1
            features = row[FEATURE_COLS].values
            inference_results = run_inference(features)
            
            payload = {
                "raw_data": [row['Raw']] * 14, # duplicate the single 'Raw' value 14 times so the brain heatmap still renders something
                "features": features.tolist(),
                "inference": inference_results,
                "ground_truth": int(row['user-definedlabeln']),
                "current_row": current_row,
                "total_rows": total_rows
            }
            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(speed_ms / 1000.0)
            
        await websocket.send_text(json.dumps({"status": "done"}))
    except Exception as e:
        print("WebSocket disconnected:", e)

@app.post("/api/upload")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Must be a CSV file")
        
    try:
        df = pd.read_csv(file.file)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid CSV format")
        
    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_cols:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing_cols[:5]}")
        
    sample_df = df.head(100)
    batch_features = sample_df[FEATURE_COLS].values
    
    results = []
    for row in batch_features:
        results.append(run_inference(row))
        
    avg_latency = {
        'rf': np.mean([r['rf']['latency_ms'] for r in results]),
        'enhanced_nn': np.mean([r['enhanced_nn']['latency_ms'] for r in results]),
        'rnn': np.mean([r['rnn']['latency_ms'] for r in results]),
        'resnet': np.mean([r['resnet']['latency_ms'] for r in results]),
        'cnn_rnn': np.mean([r['cnn_rnn']['latency_ms'] for r in results]),
    }
    
    summary = json.dumps(avg_latency)
    conn = sqlite3.connect("history.db")
    c = conn.cursor()
    c.execute("INSERT INTO upload_logs (filename, upload_time, num_rows, summary) VALUES (?, ?, ?, ?)",
              (file.filename, datetime.now().isoformat(), len(df), summary))
    conn.commit()
    conn.close()
    
    return {"message": "Success", "processed_rows": len(sample_df), "avg_latency": avg_latency}

@app.get("/api/sample")
def get_sample():
    if df_full is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded")
    sample_df = df_full.head(100)
    batch_features = sample_df[FEATURE_COLS].values
    results = []
    for row in batch_features:
        results.append(run_inference(row))
    return {"message": "Success", "results": results}

@app.get("/api/valid-sessions")
def get_valid_sessions():
    if df_full is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded")
    
    grouped = df_full.groupby('SubjectID')['VideoID'].unique()
    valid_sessions = {str(int(k)): [int(x) for x in v.tolist()] for k, v in grouped.items()}
    return valid_sessions

@app.get("/api/sample-schema")
def get_sample_schema():
    if df_full is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded")
    sample_row = df_full[FEATURE_COLS].iloc[0].to_dict()
    return sample_row

@app.get("/api/history")
def get_history():
    conn = sqlite3.connect("history.db")
    c = conn.cursor()
    c.execute("SELECT * FROM upload_logs ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    return [{"id": r[0], "filename": r[1], "upload_time": r[2], "num_rows": r[3], "summary": json.loads(r[4])} for r in rows]

@app.delete("/api/history/{log_id}")
def delete_history(log_id: int):
    conn = sqlite3.connect("history.db")
    c = conn.cursor()
    c.execute("DELETE FROM upload_logs WHERE id=?", (log_id,))
    conn.commit()
    conn.close()
    return {"message": "Deleted"}
