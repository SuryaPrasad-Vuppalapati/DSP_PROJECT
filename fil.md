# Low-Level Design (LLD) — Cloud Outage Prediction System

**Date**: February 25, 2026  
**Scope**: Current repository contents (FastAPI model service, Streamlit webapp, PostgreSQL, data scripts, Docker Compose).  
**Out of scope**: Airflow, Great Expectations, MLflow, Grafana (not implemented in this repo).

---

## 1) System Overview

The system provides:
- **On-demand predictions** via a Streamlit UI
- **Prediction API** via FastAPI
- **Persistence** of predictions to PostgreSQL
- **Data preparation utilities** (split + data error injection)

### Runtime Topology (Defense 1 scope)
- **webapp** → HTTP calls to **api**
- **api** → SQLAlchemy ORM → **db**
- **api** → reads dataset from `/data/cloud_outages_dataset.csv` and model from `/data/model.pkl`

---

## 2) Component Details

### 2.1 FastAPI Model Service
**Location**: model_service/app

#### Files
- main.py — API endpoints + app lifecycle
- model.py — training, loading, inference pipeline
- database.py — SQLAlchemy engine, session, ORM model
- schemas.py — Pydantic request/response models

#### Lifecycle
- On startup (`lifespan`):
  - Create DB tables via `Base.metadata.create_all(bind=engine)`
  - Attempt to load model from `/data/model.pkl`
  - If missing and dataset exists, train from `/data/cloud_outages_dataset.csv`

#### API Endpoints
1. **GET /health**
   - Response: `{ status: "healthy", model_loaded: boolean }`

2. **POST /predict**
   - Request: `BatchPredictionRequest`
   - Body: `{ features: [ { ... } ], source: "webapp"|"scheduled" }`
   - For each feature set:
     - Inference → predicted hours
     - `is_anomaly = predicted_hours > 5.0`
     - Persist prediction record
   - Response: `BatchPredictionResponse`

3. **GET /past-predictions**
   - Query params: `start_date`, `end_date`, `source`, `limit`
   - Returns array of `PastPrediction`

#### Data Flow (Predict)
1. Request validated by Pydantic
2. `predict()` in model.py uses scikit-learn pipeline
3. Result saved in PostgreSQL
4. API returns prediction + model version

#### Errors
- Model missing → HTTP 503
- Prediction failure → HTTP 500

---

### 2.2 Model Training & Inference
**Location**: model_service/app/model.py

#### Features
- Categorical: cloud_provider, service, severity
- Numeric: system_load_before_outage, number_of_customers_affected, ticket_count
- Derived: backup_system_triggered_enc

#### Pipeline
- FunctionTransformer to encode backup
- ColumnTransformer:
  - StandardScaler for numeric features
  - OneHotEncoder for categorical features
- RandomForestRegressor (100 trees)

#### Target
- duration_minutes converted to hours

#### Artifacts
- Model saved to `/data/model.pkl`

---

### 2.3 Database
**Location**: model_service/app/database.py, db/init.sql

#### Connection
- `DATABASE_URL` from `.env`

#### Tables
1. predictions
   - id, timestamp, model_version, source
   - features: cloud_provider, service, severity, system_load_before_outage,
     number_of_customers_affected, ticket_count, backup_system_triggered
   - outputs: predicted_hours, is_anomaly

2. ingestion_stats (placeholder for future ingestion pipeline)
   - timestamp, filename, row counts, criticality, error_types

---

### 2.4 Streamlit Webapp
**Location**: webapp/

#### Pages
1. 1_Prediction.py
   - Single prediction form
   - Batch prediction (CSV upload)
   - Calls `POST /predict`

2. pages/2_Past_Predictions.py
   - Filters: start_date, end_date, source, limit
   - Calls `GET /past-predictions`
   - Shows table + summary metrics

#### API Integration
- API base URL from `API_URL` environment variable

---

### 2.5 Data Utilities
**Location**: split_dataset.py, generate_data_issues.py

1. split_dataset.py
   - Splits dataset into N files in `data/raw_data`

2. generate_data_issues.py
   - Injects 7 error types (required + additional)
   - Outputs corrupted CSV for ingestion testing

---

## 3) Deployment & Runtime

### 3.1 Docker Compose
**File**: docker-compose.yml

Services:
- db: Postgres 15
- api: FastAPI service
- webapp: Streamlit

Key details:
- Health checks for db + api
- Shared volume: `postgres_data`
- Environment from `.env`

### 3.2 Dockerfiles
- model_service/Dockerfile
  - Installs root requirements
  - Runs `uvicorn app.main:app`

- webapp/Dockerfile
  - Installs root requirements
  - Runs `streamlit run 1_Prediction.py`

---

## 4) Data Contracts

### 4.1 Prediction Request
```json
{
  "features": [
    {
      "cloud_provider": "AWS",
      "service": "Compute",
      "severity": "High",
      "system_load_before_outage": 75,
      "number_of_customers_affected": 1200,
      "ticket_count": 35,
      "backup_system_triggered": "Yes"
    }
  ],
  "source": "webapp"
}
```

### 4.2 Prediction Response
```json
{
  "predictions": [
    {
      "cloud_provider": "AWS",
      "service": "Compute",
      "severity": "High",
      "system_load_before_outage": 75,
      "number_of_customers_affected": 1200,
      "ticket_count": 35,
      "backup_system_triggered": "Yes",
      "predicted_hours": 2.41,
      "is_anomaly": false,
      "model_version": "v1.0"
    }
  ]
}
```

---

## 5) Security & Configuration
- Secrets managed in `.env` (database credentials)
- DB URL injected via `DATABASE_URL`
- No direct DB access from webapp

---

## 6) Known Gaps (Compared to Full DSP Spec)
- No Airflow DAGs yet
- No Great Expectations ingestion validation yet
- No MLflow model registry
- No Grafana monitoring
- CI/CD pipeline not present in repo

---

## 7) Suggested Next Steps
- Add Airflow DAGs for ingestion and prediction
- Implement Great Expectations validation + data docs
- Add MLflow training pipeline and model registry
- Add Grafana dashboards for drift and data quality
- Add CI pipeline (flake8 + pytest)
