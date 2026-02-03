import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os

# Parse hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument("--test_size", type=float, default=0.25)
args = parser.parse_args()

# Fake dataset
X = pd.DataFrame({'size': [50, 60, 70, 80], 'bedrooms': [1, 2, 2, 3]})
y = np.array([150, 200, 250, 300])

# Train/test split using hyperparameter
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Test prediction
preds = model.predict(X_test)
print("Predictions:", preds)
print("MSE:", mean_squared_error(y_test, preds))

# Save model like SageMaker
model_dir = "model"
os.makedirs(model_dir, exist_ok=True)
joblib.dump(model, os.path.join(model_dir, "model.joblib"))
print("Model saved in", model_dir)
