from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert "message" in response.json()


def test_predict():
    sample_customer = {
        "tenure_months": 6,
        "support_tickets": 3,
        "monthly_spend_inr": 799,
        "last_login_days": 14,
        "plan_type": "Basic"
    }

    response = client.post(
        "/predict",
        json=sample_customer
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "churn_probability" in data
    assert "non_churn_probability" in data

    assert data["prediction"] in [0, 1]

    assert 0 <= data["churn_probability"] <= 1
    assert 0 <= data["non_churn_probability"] <= 1


def test_invalid_input():
    response = client.post(
        "/predict",
        json={"tenure_months": "invalid"}
    )

    assert response.status_code == 422
