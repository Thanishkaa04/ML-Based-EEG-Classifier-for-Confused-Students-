import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
df = pd.read_csv("EEG_data.csv")
features = ['Attention', 'Mediation', 'Raw', 'Delta', 'Theta', 'Alpha1', 'Alpha2', 'Beta1', 'Beta2', 'Gamma1', 'Gamma2']

print("1. Adding Theta/Beta and other neuro-marker ratios...")
epsilon = 1e-8
df['Theta_Beta_Ratio'] = df['Theta'] / (df['Beta1'] + df['Beta2'] + epsilon)
df['Theta_Alpha_Ratio'] = df['Theta'] / (df['Alpha1'] + df['Alpha2'] + epsilon)
df['Delta_Theta_Ratio'] = df['Delta'] / (df['Theta'] + epsilon)

print("2. Applying sliding window aggregates (rolling mean & std)...")
def add_rolling_features(group):
    window = 5
    for col in features:
        group[f'{col}_mean_5'] = group[col].rolling(window=window, min_periods=1).mean()
        group[f'{col}_std_5'] = group[col].rolling(window=window, min_periods=1).std().fillna(0)
    return group

df_engineered = df.groupby(['SubjectID', 'VideoID']).apply(add_rolling_features).reset_index(drop=True)

base_features = features + ['Theta_Beta_Ratio', 'Theta_Alpha_Ratio', 'Delta_Theta_Ratio']
rolling_features = [f"{col}_mean_5" for col in features] + [f"{col}_std_5" for col in features]
final_features = base_features + rolling_features

X = df_engineered[final_features].values
y = df_engineered['user-definedlabeln'].values

print(f"Total Features: {len(final_features)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Random Forest with new features...")
rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

preds = rf.predict(X_test)
acc = accuracy_score(y_test, preds)

print(f"Random Forest Accuracy with Feature Engineering: {acc*100:.2f}%")

