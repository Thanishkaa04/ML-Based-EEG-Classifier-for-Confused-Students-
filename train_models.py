import pandas as pd
import numpy as np
import os
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import sys
sys.path.append('backend')
from models import EnhancedNN, RNNModel, ResNet, CNN_RNN

os.makedirs("backend/models", exist_ok=True)

df = pd.read_csv("EEG_data.csv")

# Feature Engineering
features = ['Attention', 'Mediation', 'Raw', 'Delta', 'Theta', 'Alpha1', 'Alpha2', 'Beta1', 'Beta2', 'Gamma1', 'Gamma2']
epsilon = 1e-8
df['Theta_Beta_Ratio'] = df['Theta'] / (df['Beta1'] + df['Beta2'] + epsilon)
df['Theta_Alpha_Ratio'] = df['Theta'] / (df['Alpha1'] + df['Alpha2'] + epsilon)
df['Delta_Theta_Ratio'] = df['Delta'] / (df['Theta'] + epsilon)

def add_rolling_features(group):
    window = 5
    for col in features:
        group[f'{col}_mean_5'] = group[col].rolling(window=window, min_periods=1).mean()
        group[f'{col}_std_5'] = group[col].rolling(window=window, min_periods=1).std().fillna(0)
    return group

df = df.groupby(['SubjectID', 'VideoID']).apply(add_rolling_features).reset_index(drop=True)

base_features = features + ['Theta_Beta_Ratio', 'Theta_Alpha_Ratio', 'Delta_Theta_Ratio']
rolling_features = [f"{col}_mean_5" for col in features] + [f"{col}_std_5" for col in features]
final_features = base_features + rolling_features

X = df[final_features].values
y = df['user-definedlabeln'].values

print(f"Training on {X.shape[1]} Engineered Features...")

from sklearn.ensemble import RandomForestClassifier

# Standard split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'backend/models/scaler.joblib')

print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
acc = accuracy_score(y_test, rf.predict(X_test_scaled))
print(f"RF Acc: {acc*100:.2f}%")
joblib.dump(rf, 'backend/models/rf_model.joblib')

device = torch.device('cpu')
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

def train_pt(model, name, epochs=10):
    print(f"Training {name}...")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            out = model(batch_X)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        out = model(X_test_t)
        preds = torch.argmax(out, dim=1)
        acc = accuracy_score(y_test_t.numpy(), preds.numpy())
        print(f"{name} Acc: {acc*100:.2f}%")
        
    torch.save(model.state_dict(), f"backend/models/{name}.pth")

enn = EnhancedNN(num_classes=2, input_feature_size=36)
train_pt(enn, "EnhancedNN", epochs=15)

rnn = RNNModel(input_size=36, hidden_size=128, output_size=2)
train_pt(rnn, "RNNModel", epochs=15)

resnet = ResNet(num_classes=2)
train_pt(resnet, "ResNet", epochs=1)

cnn_rnn = CNN_RNN(num_classes=2)
train_pt(cnn_rnn, "CNN_RNN", epochs=1)

print("Training complete!")
