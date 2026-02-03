# SageMaker Dry-Run Project 
This project simulates a full SageMaker workflow locally. 
 
## Project Structure 
- `train.py`: simple training script 
- `train_sagemaker_style.py`: training script with hyperparameters (SageMaker-style) 
- `save_model.py`: saves trained model as SageMaker artifact (`model.tar.gz`) 
- `predict.py`: simulates deployment and inference locally 
- `.gitignore`: ignores model files and cache 
 
## How to Run 
1. Create Conda environment: 
`conda create -n sagemaker_dryrun python=3.11` 
2. Activate environment: 
`conda activate sagemaker_dryrun` 
3. Install packages: 
`pip install numpy pandas scikit-learn joblib` 
4. Train model: 
`python train_sagemaker_style.py --test_size 0.5` 
5. Simulate deployment: 
`python predict.py` 
 
## Notes 
- Model files (`model/` and `model.tar.gz`) are ignored in Git. 
- This project is fully local and $0 cost. 
