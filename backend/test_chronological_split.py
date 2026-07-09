import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def main():
    print("Loading full dataset...")
    df = pd.read_csv("../../Kaggle_eeg_data.csv")
    
    # We assume the dataset is in chronological order per session.
    # We will iterate through each subject, and take the first 80% of their rows for training, 
    # and the last 20% of their rows for testing.
    
    train_indices = []
    test_indices = []
    
    # Group by subject_id
    for subject_id, group in df.groupby('subject_id'):
        n_rows = len(group)
        split_point = int(n_rows * 0.8)
        
        # We append the original dataframe index to keep track
        train_indices.extend(group.index[:split_point].tolist())
        test_indices.extend(group.index[split_point:].tolist())
        
    print(f"Total Train rows (First 80% per subject): {len(train_indices)}")
    print(f"Total Test rows (Last 20% per subject): {len(test_indices)}")
    
    feature_cols = df.columns[2:86]
    X_train_raw = df.loc[train_indices, feature_cols].values
    y_train = df.loc[train_indices, 'video_id'].values
    
    X_test_raw = df.loc[test_indices, feature_cols].values
    y_test = df.loc[test_indices, 'video_id'].values
    
    print("Standardizing features (fit on train only to prevent leakage)...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    
    # 1. Logistic Regression
    print("\nTraining Logistic Regression on Chronological Split...")
    lr_model = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, max_iter=50)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    acc = accuracy_score(y_test, lr_pred)
    
    print(f"Logistic Regression Chronological Accuracy: {acc * 100:.2f}%")
    # Prepare Tensors for PyTorch
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    from models import EnhancedNN, RNNModel, ResNet, CNN_RNN
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    
    device = torch.device("cpu")
    num_classes = 11
    
    def train_and_eval(model, name, epochs=3):
        print(f"\nTraining {name} on Chronological Split for {epochs} epochs...")
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        
        for epoch in range(epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_t.to(device))
            _, predicted = torch.max(outputs.data, 1)
            acc = accuracy_score(y_test_t.cpu().numpy(), predicted.cpu().numpy())
            print(f"{name} Chronological Accuracy: {acc * 100:.2f}%")
            
    enhanced_nn = EnhancedNN(num_classes=num_classes, input_feature_size=84)
    train_and_eval(enhanced_nn, "EnhancedNN")
    
    rnn_model = RNNModel(input_size=14, hidden_size=128, output_size=num_classes)
    train_and_eval(rnn_model, "RNNModel")
    
    resnet_model = ResNet(num_classes=num_classes)
    train_and_eval(resnet_model, "ResNet", epochs=2)
    
    cnn_rnn_model = CNN_RNN(num_classes=num_classes)
    train_and_eval(cnn_rnn_model, "CNN_RNN", epochs=2)

    print("\nTest complete.")

if __name__ == "__main__":
    main()
