import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

from models import EnhancedNN, RNNModel, ResNet, CNN_RNN

def main():
    print("Loading data...")
    csv_path = "../../Kaggle_eeg_data.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path).sample(n=10000, random_state=42)
    
    feature_cols = df.columns[2:86]
    X = df[feature_cols].values
    y = df['video_id'].values
    
    print("Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.joblib")
    print("Saved StandardScaler.")
    
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, max_iter=200)
    lr_model.fit(X_scaled, y)
    joblib.dump(lr_model, "models/lr_model.joblib")
    print("Saved Logistic Regression.")
    
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    device = torch.device("cpu")
    num_classes = len(np.unique(y))
    num_classes = 11
    
    def train_pytorch_model(model, name, epochs=10, lr=0.001):
        print(f"Training {name} for {epochs} epochs...")
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"{name} Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")
            
        torch.save(model.state_dict(), f"models/{name}.pth")
        print(f"Saved {name}.")
    
    enhanced_nn = EnhancedNN(num_classes=num_classes, input_feature_size=84)
    train_pytorch_model(enhanced_nn, "EnhancedNN", epochs=10)
    
    rnn_model = RNNModel(input_size=14, hidden_size=128, output_size=num_classes)
    train_pytorch_model(rnn_model, "RNNModel", epochs=10)
    
    resnet_model = ResNet(num_classes=num_classes)
    train_pytorch_model(resnet_model, "ResNet", epochs=2)
    
    cnn_rnn_model = CNN_RNN(num_classes=num_classes)
    train_pytorch_model(cnn_rnn_model, "CNN_RNN", epochs=10)
    
    print("All models trained and saved successfully.")

if __name__ == "__main__":
    main()
