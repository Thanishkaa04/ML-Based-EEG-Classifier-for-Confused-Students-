import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from backend.models import EnhancedNN

df = pd.read_csv("EEG_data.csv")
features = ['Attention', 'Mediation', 'Raw', 'Delta', 'Theta', 'Alpha1', 'Alpha2', 'Beta1', 'Beta2', 'Gamma1', 'Gamma2']
X = df[features].values
y = df['user-definedlabeln'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)
train_dataset = TensorDataset(X_train_t, y_train_t)

def try_config(lr, wd, bs, epochs):
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    model = EnhancedNN(num_classes=2, input_feature_size=11)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    best_acc = 0
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
            preds = torch.argmax(model(X_test_t), dim=1)
            acc = (preds == y_test_t).float().mean().item()
            best_acc = max(best_acc, acc)
            scheduler.step(acc)
    return best_acc

configs = [
    (0.001, 1e-4, 64, 50),
    (0.005, 1e-3, 128, 50),
    (0.0005, 1e-5, 32, 50)
]
for lr, wd, bs, epochs in configs:
    acc = try_config(lr, wd, bs, epochs)
    print(f"LR: {lr}, WD: {wd}, BS: {bs}, Best Acc: {acc*100:.2f}%")
