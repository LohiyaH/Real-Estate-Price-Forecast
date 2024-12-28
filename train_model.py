import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

# Load the data
df = pd.read_csv("train.csv")

# Prepare features
posted_map = {'Owner': 0, 'Dealer': 1, 'Builder': 2}

# Map categorical variables
df['POSTED_BY'] = df['POSTED_BY'].map(posted_map)

# Use LabelEncoder for location
le = LabelEncoder()
df['ADDRESS'] = le.fit_transform(df['ADDRESS'])

# Select features
features = ['POSTED_BY', 'UNDER_CONSTRUCTION', 'RERA', 'BHK_NO.', 'SQUARE_FT', 'READY_TO_MOVE', 'ADDRESS']
X = df[features]
y = df['TARGET(PRICE_IN_LACS)']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save the model, scaler, and label encoder
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("Model trained and saved successfully!")
print(f"Test score: {model.score(X_test_scaled, y_test):.4f}")

# Print feature importances
importances = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
})
print("\nFeature Importances:")
print(importances.sort_values('importance', ascending=False))
