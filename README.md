# ML-Based EEG Classifier Dashboard

This platform is a comprehensive machine learning pipeline and real-time dashboard designed to classify binary cognitive states (Not Confused vs. Confused) from single-channel EEG signals.

## Overview
This platform streams simulated live EEG data through a fully-crossed, 11-feature NeuroSky dataset. It passes this data through five distinct machine learning models concurrently (including PyTorch deep learning networks and a Scikit-Learn Random Forest ensemble) and dynamically renders confidence probabilities, inference latencies, and binary confusion matrices in real time.

### Key Features
- **Real-Time WebSocket Inference:** Streams single-channel EEG inputs to 5 independent ML models simultaneously, computing real-time confidence scores and inference latencies.
- **Advanced Feature Engineering:** Computes a 36-dimensional feature vector in-memory via sliding window temporals (Mean/Std) and neuro-marker ratios (Theta/Beta) to overcome single-sensor information limits.
- **Model Ensembles:** Includes CNN-RNN, ResNet, feedforward NNs, and an optimized Random Forest (achieving ~79.3% accuracy).
- **Custom Upload Validation:** A dedicated pipeline allowing researchers to push unseen EEG CSV datasets to the backend for blind inference.

## Architecture
- **Backend:** FastAPI, PyTorch, Scikit-Learn, WebSockets, Pandas (Port 8001)
- **Frontend:** React, Vite, Lucide-React (Port 5174)

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd ML-Based-EEG-Classifier-for-Confused-Students-
   ```

2. **Start the Backend:**
   ```bash
   cd backend
   pip install -r requirements.txt
   uvicorn main:app --port 8001 --reload
   ```

3. **Start the Frontend:**
   ```bash
   cd frontend
   npm install
   npm run dev -- --port 5174
   ```

4. **Run!**
   Navigate to `http://localhost:5174/` to view the live dashboard!
