# AWS SageMaker Machine Learning Project

## **Project Overview**

This project demonstrates a complete Amazon SageMaker-style machine learning workflow executed locally. It simulates model training, artifact generation, and inference to understand the end-to-end machine learning lifecycle before deploying workloads to AWS SageMaker.

---

## **Project Goal**

- Simulate the SageMaker machine learning workflow locally
- Train and evaluate a machine learning model
- Generate deployment-ready model artifacts
- Perform local inference
- Understand the SageMaker development lifecycle

---

## **Architecture**

See: `diagrams/architecture.md`

---

## **Technologies**

- AWS SageMaker Concepts
- Python
- NumPy
- Pandas
- Scikit-learn
- Joblib

---

## **Features**

- Local SageMaker workflow simulation
- Machine learning model training
- Hyperparameter configuration
- Model artifact generation
- Local inference testing
- AWS-ready project structure

---

## **Project Structure**

```text
.
├── train.py
├── train_sagemaker_style.py
├── save_model.py
├── predict.py
├── diagrams/
│   └── architecture.md
├── LICENSE
├── .gitignore
└── README.md
```

---

## **Deployment**

Create the Conda environment:

```bash
conda create -n sagemaker_dryrun python=3.11
```

Activate the environment:

```bash
conda activate sagemaker_dryrun
```

Install dependencies:

```bash
pip install numpy pandas scikit-learn joblib
```

Train the model:

```bash
python train_sagemaker_style.py --test_size 0.5
```

Run inference:

```bash
python predict.py
```

---

## **Key Learnings**

- Machine learning workflow fundamentals
- SageMaker-style training pipelines
- Model artifact generation
- Local model inference
- Preparing workloads for AWS SageMaker

---

## **Status**

Completed

---

## **Future Improvements**

- Deploy to Amazon SageMaker
- Hyperparameter tuning jobs
- Model registry integration
- SageMaker Pipelines
- CloudWatch monitoring
