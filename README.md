# ML Churn Prediction FastAPI

## Project Overview

This project is a Real-Time Machine Learning REST API for predicting customer churn.

The system uses a trained Random Forest classification model and FastAPI to provide real-time churn predictions and prediction probabilities.

## Problem Statement

Customer churn is an important business problem. The goal of this project is to predict whether a customer is likely to churn based on customer information such as tenure, support tickets, monthly spending, last login activity, and plan type.

## Machine Learning Features

The model uses the following input features:

- tenure_months
- support_tickets
- monthly_spend_inr
- last_login_days
- plan_type

Target:

- churned
- 0 = Non-Churn
- 1 = Churn

## Model

Multiple machine learning models were evaluated during model development.

The final trained model is stored as:

`champion_churn_model.joblib`

The model includes preprocessing and classification steps so that incoming customer data can be directly processed for prediction.

## REST API

The machine learning model is served using FastAPI.

Main endpoint:

`POST /predict`

Example JSON request:

```json
{
  "tenure_months": 6,
  "support_tickets": 3,
  "monthly_spend_inr": 799,
  "last_login_days": 14,
  "plan_type": "Basic"
}
```

Example response structure:

```json
{
  "prediction": 1,
  "churn_probability": 0.85,
  "non_churn_probability": 0.15
}
```

The probability values depend on the model prediction.

## System Architecture

Customer Data  
↓  
JSON Request  
↓  
FastAPI REST API  
↓  
Input Validation using Pydantic  
↓  
Machine Learning Preprocessing Pipeline  
↓  
Champion Churn Prediction Model  
↓  
Prediction and Probability  
↓  
JSON Response

## Project Files

- `main.py` - FastAPI application and prediction endpoint
- `champion_churn_model.joblib` - Trained machine learning model
- `requirements.txt` - Python dependencies
- `Dockerfile` - Container configuration
- `test_main.py` - API unit tests
- `README.md` - Project documentation

## Docker

Build the Docker image:

```bash
docker build -t churn-api .
```

Run the container:

```bash
docker run -p 8000:8000 churn-api
```

## API Testing

Unit tests validate:

- API response status codes
- Prediction response schema
- Churn prediction output
- Prediction probability ranges
- Invalid input handling

Run tests using:

```bash
pytest test_main.py
```

## Technologies Used

- Python
- Scikit-learn
- FastAPI
- Pandas
- Joblib
- Pydantic
- Docker
- Pytest

## Conclusion

This project demonstrates an end-to-end machine learning deployment workflow. It includes model training, preprocessing, model serialization, a real-time FastAPI prediction service, Docker containerization, API testing, and system documentation.
