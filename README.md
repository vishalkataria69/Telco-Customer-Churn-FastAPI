# Telco Customer Churn Prediction Dashboard

A full-stack Machine Learning application that predicts whether a telecom customer is likely to **churn or stay**, built with a **FastAPI** prediction backend and an interactive **Streamlit** frontend.

---

## Project Overview

Customer churn is one of the biggest challenges in the telecom industry. This project delivers an end-to-end ML pipeline — from data exploration and model training to real-time predictions served via a REST API and visualized through an interactive dashboard.

| Attribute         | Detail                          |
|-------------------|---------------------------------|
| **Algorithm**     | Logistic Regression             |
| **Problem Type**  | Binary Classification           |
| **Dataset**       | 7,043 Telco customers           |
| **Features**      | 19                              |
| **Accuracy**      | ~82%                            |
| **Scaling**       | StandardScaler                  |
| **Backend**       | FastAPI + Uvicorn               |
| **Frontend**      | Streamlit                       |

---

## Features

### Streamlit Dashboard

| Page | Description |
|------|-------------|
| **Home** | Project overview, KPI cards, ML workflow diagram |
| **Prediction** | Interactive form -> real-time churn prediction with probability gauge, risk level & business recommendations |
| **Data Insights** | Interactive charts, filterable dataset, correlation heatmap, downloadable CSV |
| **AI Retention Assistant** | Rule-based recommendation engine, targeted offer generator, suggested customer message, chatbot interface |
| **Model Performance** | Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix, Feature Importance |
| **About** | Developer profile, skills, certifications, and achievements |

### FastAPI Backend

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | `GET` | API root info |
| `/api/v1/health` | `GET` | Health check — confirms model is loaded |
| `/api/v1/model-info` | `GET` | Algorithm, feature list, test set size |
| `/api/v1/predict` | `POST` | Single customer churn prediction |
| `/api/v1/predict/batch` | `POST` | Batch churn predictions |
| `/api/v1/metrics` | `GET` | Full model performance metrics + ROC curve data |
| `/api/v1/feature-importance` | `GET` | Sorted feature importance (model coefficients) |
| `/api/v1/classification-report` | `GET` | Full classification report |

---

## Architecture

```
+----------------------------------+
|         Streamlit App            |
|  (app.py + app_pages/*.py)       |
|                                  |
|  Home  *  Prediction             |
|  Data Insights                   |
|  AI Retention Assistant          |
|  Model Performance  *  About     |
+---------------+------------------+
                |  HTTP (via ChurnAPIClient)
                v
+----------------------------------+
|       FastAPI Backend            |
|  (api.py -- Uvicorn server)      |
|                                  |
|  * Input validation (Pydantic)   |
|  * Feature encoding              |
|  * StandardScaler transform      |
|  * Logistic Regression predict   |
|  * Risk & Recommendation engine  |
+---------------+------------------+
                |
                v
+----------------------------------+
|   models/*.pkl                   |
|                                  |
|  * model (LogisticRegression)    |
|  * scaler (StandardScaler)       |
|  * feature_columns               |
|  * x_test / y_test               |
+----------------------------------+
```

---

## Project Structure

```
Telco-Customer-Churn-FastAPI/
|
+-- api.py                   # FastAPI application (all routes & schemas)
+-- api_client.py            # Thin HTTP client used by the Streamlit app
+-- app.py                   # Streamlit entry point + sidebar navigation
+-- prediction.py            # Prediction page (calls API via ChurnAPIClient)
+-- requirements.txt         # Python dependencies
|
+-- app_pages/
|   +-- Home.py              # Landing page with overview & ML workflow
|   +-- Data_Insights.py     # Interactive EDA charts & filtered dataset
|   +-- Model_Performance.py # Metrics, confusion matrix, ROC, feature importance
|   +-- Retention_Assistant.py  # AI recommendation engine & chatbot
|   +-- about.py             # Developer profile page
|
+-- models/
|   +-- Telco_customer_churn_model_and_preprocessors.pkl
|
+-- data/
|   +-- Telco-Customer-Churn.csv   # Source dataset (7,043 rows)
|
+-- notebook/
|   +-- Telco_Customer_Churn.ipynb # EDA & model training notebook
|
+-- assets/
    +-- image_2.png          # Sidebar banner image
    +-- formal image.png     # Developer profile photo
```

---

## Installation & Setup

### Prerequisites

- Python 3.9+
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Telco-Customer-Churn-FastAPI.git
cd Telco-Customer-Churn-FastAPI
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

The application requires **two processes** running concurrently.

### Terminal 1 — Start the FastAPI Backend

```bash
uvicorn api:app --reload --port 8000
```

Once running:
- **Base URL:** `http://127.0.0.1:8000`
- **Swagger UI:** `http://127.0.0.1:8000/docs`
- **ReDoc:** `http://127.0.0.1:8000/redoc`

### Terminal 2 — Start the Streamlit Frontend

```bash
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`.

> **Note:** The FastAPI backend **must be running first**. The Prediction page will show an error and stop rendering if the API is unreachable.

---

## API Usage Examples

### Single Prediction

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "senior_citizen": 0,
    "partner": "Yes",
    "dependents": "No",
    "tenure": 12,
    "phone_service": "Yes",
    "multiple_lines": "No",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "online_backup": "Yes",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "Yes",
    "streaming_movies": "No",
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
    "monthly_charges": 70.35,
    "total_charges": 845.5
  }'
```

**Sample Response:**

```json
{
  "churn_prediction": 1,
  "churn_label": "Churn",
  "churn_probability": 0.7823,
  "risk_level": "High",
  "recommendation": [
    "Immediate retention campaign",
    "Special discount",
    "Dedicated relationship manager",
    "Priority customer support"
  ]
}
```

### Health Check

```bash
curl http://127.0.0.1:8000/api/v1/health
```

```json
{ "status": "ok", "model_loaded": true }
```

### Batch Prediction

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{"customers": [ {...customer1...}, {...customer2...} ]}'
```

---

## Risk Classification

| Churn Probability | Risk Level | Action |
|-------------------|------------|--------|
| < 30%             | Low        | Continue current plan, loyalty rewards |
| 30% - 60%         | Medium     | Offer annual contract, personalized offers |
| > 60%             | High       | Immediate retention campaign, special discount |

---

## Model Details

- **Algorithm:** Logistic Regression (scikit-learn)
- **Scaling:** StandardScaler on all 19 features
- **Serialization:** Pickle `.pkl` — stores model, scaler, feature columns, and held-out test set
- **Feature Encoding:** Manual ordinal encoding (identical to training-time encoding)
- **Notebook:** `notebook/Telco_Customer_Churn.ipynb` — full EDA and training walkthrough

### Input Features

| Feature | Type | Allowed Values |
|---------|------|----------------|
| gender | Categorical | Female, Male |
| senior_citizen | Integer | 0, 1 |
| partner | Categorical | Yes, No |
| dependents | Categorical | Yes, No |
| tenure | Integer | 0 - 100 |
| phone_service | Categorical | Yes, No |
| multiple_lines | Categorical | No, Yes, No phone service |
| internet_service | Categorical | DSL, Fiber optic, No |
| online_security | Categorical | No, Yes, No internet service |
| online_backup | Categorical | No, Yes, No internet service |
| device_protection | Categorical | No, Yes, No internet service |
| tech_support | Categorical | No, Yes, No internet service |
| streaming_tv | Categorical | No, Yes, No internet service |
| streaming_movies | Categorical | No, Yes, No internet service |
| contract | Categorical | Month-to-month, One year, Two year |
| paperless_billing | Categorical | Yes, No |
| payment_method | Categorical | Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic) |
| monthly_charges | Float | >= 0 |
| total_charges | Float | >= 0 |

---

## Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Core programming language |
| FastAPI | REST API backend |
| Uvicorn | ASGI server |
| Pydantic v2 | Request/response validation & schemas |
| Streamlit | Interactive web dashboard |
| Scikit-Learn | Logistic Regression + StandardScaler |
| Pandas | Data processing |
| NumPy | Numerical computing |
| Plotly | Interactive charts & visualizations |
| Pickle | Model serialization |
| Requests | HTTP client (Streamlit -> API) |

---

## Key Business Insights

From the dataset analysis:

- Customers on **Month-to-Month contracts** churn significantly more often.
- **Fiber Optic** internet users show a higher churn rate.
- Customers with **shorter tenure** (< 12 months) are at greater risk.
- Higher **Monthly Charges** correlate with increased churn risk.
- Migrating customers to **One-Year or Two-Year contracts** can meaningfully reduce churn.

---

## Developer

**Vishal Kataria**
AI Engineer | Data Scientist | Machine Learning Engineer

Passionate about solving real-world business problems using AI, Machine Learning, Deep Learning, NLP, and Generative AI.

---

## License

This project is intended for educational and portfolio purposes.

---

<div align="center">

**If you found this project useful, please consider giving it a star!**

Built with love by **Vishal Kataria**

</div>
