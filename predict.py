import joblib
import pandas as pd
import tarfile
import os

# Extract model.tar.gz like SageMaker would
with tarfile.open("model.tar.gz", "r:gz") as tar:
    tar.extractall(path="model_extracted")

# Load the model
model = joblib.load(os.path.join("model_extracted", "model", "model.joblib"))

# New data for prediction
new_data = pd.DataFrame({'size': [65, 90], 'bedrooms': [2, 3]})
predictions = model.predict(new_data)

print("Predictions for new data:", predictions)
