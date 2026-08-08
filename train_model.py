import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

# Use the Heart Failure Clinical Records dataset structure
# Downloading via a simple synthetic generation matching real dataset stats
np.random.seed(42)
n = 299

data = pd.DataFrame({
    'age': np.random.normal(60.8, 11.9, n).clip(40, 95),
    'anaemia': np.random.binomial(1, 0.43, n),
    'creatinine_phosphokinase': np.random.exponential(581, n).clip(23, 7861),
    'diabetes': np.random.binomial(1, 0.42, n),
    'ejection_fraction': np.random.normal(38.1, 11.8, n).clip(14, 80),
    'high_blood_pressure': np.random.binomial(1, 0.35, n),
    'platelets': np.random.normal(263358, 97804, n).clip(25100, 850000),
    'serum_creatinine': np.random.exponential(1.39, n).clip(0.5, 9.4),
    'serum_sodium': np.random.normal(136.6, 4.4, n).clip(113, 148),
    'sex': np.random.binomial(1, 0.65, n),
    'smoking': np.random.binomial(1, 0.32, n),
    'time': np.random.uniform(4, 285, n),
})

# Target: higher risk with low ejection_fraction, high serum_creatinine, older age
risk = (
    (data['ejection_fraction'] < 35).astype(int) +
    (data['serum_creatinine'] > 1.5).astype(int) +
    (data['age'] > 65).astype(int) +
    (data['time'] < 50).astype(int)
)
data['DEATH_EVENT'] = (risk >= 2).astype(int)

X = data.drop('DEATH_EVENT', axis=1)
y = data['DEATH_EVENT']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

with open('heart_failure_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model and scaler saved successfully.")
