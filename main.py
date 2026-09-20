from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Real-Time ML API for predicting customer churn",
    version="1.0"
)

# Load trained champion model
model = joblib.load("champion_churn_model.joblib")


class CustomerData(BaseModel):
    tenure_months: int
    support_tickets: int
    monthly_spend_inr: float
    last_login_days: int
    plan_type: str


@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running"
    }


@app.post("/predict")
def predict(data: CustomerData):

    customer = pd.DataFrame([{
        "tenure_months": data.tenure_months,
        "support_tickets": data.support_tickets,
        "monthly_spend_inr": data.monthly_spend_inr,
        "last_login_days": data.last_login_days,
        "plan_type": data.plan_type
    }])

    prediction = model.predict(customer)[0]

    probabilities = model.predict_proba(customer)[0]

    return {
        "prediction": int(prediction),
        "churn_probability": float(probabilities[1]),
        "non_churn_probability": float(probabilities[0])
    }
