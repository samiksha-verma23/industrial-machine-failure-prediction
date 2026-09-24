# Industrial Machine Failure Prediction

A full-stack machine learning application that predicts industrial machine failure using operating conditions from the **AI4I 2020 Predictive Maintenance Dataset**.

## Features

* Machine failure prediction
* Random Forest ML model
* Flask REST API backend
* HTML, CSS & JavaScript frontend
* Failure probability prediction
* Prediction history
* Feature importance
* Model performance dashboard

## Tech Stack

**Machine Learning:** Python, Pandas, NumPy, Scikit-learn, Joblib
**Backend:** Flask, Flask-CORS
**Frontend:** HTML, CSS, JavaScript

## Model Performance

| Model               |   Accuracy |   F1 Score |    ROC-AUC |
| ------------------- | ---------: | ---------: | ---------: |
| Logistic Regression |     82.45% |     24.19% |     90.70% |
| Decision Tree       |     95.35% |     54.63% |     90.06% |
| Random Forest       | **97.30%** | **63.01%** | **97.11%** |

Random Forest was selected based on F1 score in the evaluation.

## Project Structure

```text
industrial-machine-failure-prediction/
├── backend/
├── frontend/
├── data/
├── models/
├── data_analysis.py
├── train_model.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Run Locally

### Backend

```bash
python backend/app.py
```

Runs at `http://127.0.0.1:5000`

### Frontend

```bash
cd frontend
python server.py
```

Runs at `http://127.0.0.1:5500`

## Dataset

**AI4I 2020 Predictive Maintenance Dataset**
Target: `Machine failure`

The model uses machine type, air temperature, process temperature, rotational speed, torque, and tool wear as input features.

## Author

**Samiksha Verma**
B.Tech CSE
