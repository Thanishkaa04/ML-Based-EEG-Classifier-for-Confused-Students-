import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from models import EnhancedNN, RNNModel, ResNet, CNN_RNN

def main():
    print("Loading full dataset...")
    df = pd.read_csv("../../Kaggle_eeg_data.csv").sample(n=20000, random_state=42)
    
    feature_cols = df.columns[2:86]
    X = df[feature_cols].values
    y = df['video_id'].values
    groups = df['subject_id'].values
    
    train_mask = np.isin(groups, [0, 1, 2, 3, 4, 5])
    test_mask = np.isin(groups, [6, 7])
    
    X_train_raw, y_train = X[train_mask], y[train_mask]
    X_test_raw, y_test = X[test_mask], y[test_mask]
    
    print(f"Train set: {len(X_train_raw)} rows. Test set: {len(X_test_raw)} rows.")
    
    print("Standardizing features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    
    print("\nTraining Logistic Regression...")
    lr_model = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, max_iter=50)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    print(f"Logistic Regression True Accuracy: {accuracy_score(y_test, lr_pred) * 100:.2f}%")
    
    print("Test complete.")

if __name__ == "__main__":
    main()
