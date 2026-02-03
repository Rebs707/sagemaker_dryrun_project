import joblib
from train import model  # import the trained model from train.py
import os
import tarfile

# Make a directory to store the model
model_dir = "model"
os.makedirs(model_dir, exist_ok=True)

# Save the model
joblib.dump(model, os.path.join(model_dir, "model.joblib"))

# Create a tar.gz archive (SageMaker format)
with tarfile.open("model.tar.gz", "w:gz") as tar:
    tar.add(model_dir, arcname=os.path.basename(model_dir))

print("Model saved as model.tar.gz")
