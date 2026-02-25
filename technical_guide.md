# Deep Technical Guide — Cloud Outage Prediction System

**Complete A-to-Z Implementation Guide**

---

## Table of Contents
1. [Project Structure Overview](#project-structure-overview)
2. [Root Level Files](#root-level-files)
3. [Model Service (FastAPI)](#model-service-fastapi)
4. [Webapp (Streamlit)](#webapp-streamlit)
5. [Database Layer](#database-layer)
6. [Data Management](#data-management)
7. [Docker Infrastructure](#docker-infrastructure)
8. [Machine Learning Pipeline](#machine-learning-pipeline)
9. [Data Flow & Architecture](#data-flow--architecture)
10. [API Contracts](#api-contracts)

---

## Project Structure Overview

```
Dsp-Project/
├── .env                          # Environment variables (secrets)
├── .env.example                  # Template for environment setup
├── .gitignore                    # Git ignore patterns
├── docker-compose.yml            # Multi-container orchestration
├── requirements.txt              # Python dependencies (shared)
├── README.md                     # Project overview
├── split_dataset.py              # Utility: split CSV into batches
├── generate_data_issues.py       # Utility: inject data quality errors
│
├── data/                         # Data storage
│   ├── cloud_outages_dataset.csv # Source training data
│   ├── model.pkl                 # Trained ML model (generated)
│   ├── raw_data/                 # Input for ingestion (batch files)
│   ├── good_data/                # Validated data
│   └── bad_data/                 # Invalid data
│
├── db/                           # Database initialization
│   └── init.sql                  # Schema creation SQL
│
├── docs/                         # Documentation
│   ├── 0_project_overview.md
│   ├── 2_prerequisite.md
│   ├── 3_defense_1.md
│   ├── low_level_design.md
│   └── deep_technical_guide.md   # This document
│
├── model_service/                # FastAPI microservice
│   ├── Dockerfile
│   └── app/
│       ├── __init__.py
│       ├── main.py               # API endpoints + lifespan
│       ├── model.py              # ML training/inference logic
│       ├── database.py           # SQLAlchemy ORM
│       └── schemas.py            # Pydantic models
│
└── webapp/                       # Streamlit frontend
    ├── Dockerfile
    ├── 1_Prediction.py           # Main prediction page
    └── pages/
        └── 2_Past_Predictions.py # Historical view
```

---

## Root Level Files

### `.env` — Environment Configuration
Contains sensitive configuration loaded by Docker Compose and application code.

```bash
DATABASE_URL=postgresql://postgres:yourpassword@db:5432/dsp
POSTGRES_USER=postgres
POSTGRES_PASSWORD=yourpassword
POSTGRES_DB=dsp
```

**Key points:**
- **DATABASE_URL**: Full connection string for SQLAlchemy (Python)
- **POSTGRES_***: Used by postgres Docker image for initialization
- **db:5432**: `db` is the Docker service name (internal DNS)
- **Never commit** `.env` to Git (use `.gitignore`)

---

### `docker-compose.yml` — Service Orchestration

Defines 3 services: `db`, `api`, `webapp`.

**Service: db (PostgreSQL)**
```yaml
db:
  image: postgres:15
  env_file: .env
  ports:
    - "5433:5432"  # Host:Container
  volumes:
    - postgres_data:/var/lib/postgresql/data  # Persist data
    - ./db/init.sql:/docker-entrypoint-initdb.d/init.sql  # Init script
  healthcheck:
    test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER}"]
```

**Why port 5433?** Host machine already has PostgreSQL on 5432. Container still uses 5432 internally.

**Volumes:**
- **Named volume** (`postgres_data`): Survives container restarts
- **Bind mount** (`./db/init.sql`): Run SQL on first startup

**Health check:** Dependent services wait until `pg_isready` succeeds.

---

**Service: api (FastAPI)**
```yaml
api:
  build:
    context: .
    dockerfile: model_service/Dockerfile
  env_file: .env
  ports:
    - "8000:8000"
  volumes:
    - ./model_service:/app  # Live code reload during dev
    - ./data:/data          # Access dataset and model
  depends_on:
    db:
      condition: service_healthy
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
```

**Build context:** Root directory (to access `requirements.txt`)  
**Dockerfile path:** `model_service/Dockerfile`  
**Volumes:** Bind mounts for development hot-reload  
**depends_on with condition:** Waits for DB to be healthy before starting

---

**Service: webapp (Streamlit)**
```yaml
webapp:
  build:
    context: .
    dockerfile: webapp/Dockerfile
  env_file: .env
  ports:
    - "8501:8501"
  environment:
    - API_URL=http://api:8000  # Container-to-container communication
  depends_on:
    api:
      condition: service_healthy
```

**API_URL:** Uses Docker service name `api`, not `localhost` (different network namespace).

---

### `requirements.txt` — Python Dependencies

Shared across all services to keep versions consistent.

```python
# Web Framework
fastapi==0.111.0          # Modern async web framework
uvicorn[standard]==0.30.1 # ASGI server with auto-reload

# Data Validation
pydantic==2.7.1           # Data validation + serialization
python-multipart==0.0.9   # For file uploads (if needed)

# Database
sqlalchemy==2.0.30        # ORM + connection pooling
psycopg2-binary==2.9.9    # PostgreSQL adapter

# ML & Data Science
pandas==2.2.2             # DataFrame operations
numpy==1.26.4             # Numerical computing
scikit-learn==1.5.0       # ML algorithms + preprocessing
joblib==1.4.2             # Model serialization

# Frontend
streamlit==1.35.0         # Web UI framework

# HTTP Client
requests==2.32.3          # API calls from Streamlit
```

**Why shared requirements?**
- Consistency: Same versions everywhere
- Simplicity: One file to maintain
- Docker layer caching: `COPY requirements.txt` → `RUN pip install`

---

### `split_dataset.py` — Data Splitting Utility

**Purpose:** Split main dataset into N files for simulating batch ingestion.

```python
def split_dataset(input_path: str, output_dir: str, num_files: int):
    df = pd.read_csv(input_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Round-robin distribution
    chunks = [df[i::num_files] for i in range(num_files)]
    
    for i, chunk in enumerate(chunks, start=1):
        out_path = os.path.join(output_dir, f"batch_{i:03d}.csv")
        chunk.to_csv(out_path, index=False)
```

**Usage:**
```bash
python split_dataset.py \
  --input data/cloud_outages_dataset.csv \
  --output data/raw_data \
  --num-files 30
```

**Key points:**
- **Round-robin slicing:** `df[0::30]`, `df[1::30]`, ... ensures even distribution
- **Filename format:** `batch_001.csv`, `batch_002.csv`, etc. (zero-padded for sorting)
- **Why needed:** Airflow ingestion DAG will read one file per execution

---

### `generate_data_issues.py` — Data Quality Error Injection

**Purpose:** Inject 7 types of data quality errors for Great Expectations testing.

```python
def inject_errors(df: pd.DataFrame, probability: float) -> pd.DataFrame:
    # 1. Completeness — Null values
    null_mask = mask()
    df.loc[null_mask, "number_of_customers_affected"] = np.nan
    
    # 2. Validity — Out of range
    range_mask = mask()
    df.loc[range_mask, "ticket_count"] = rng.integers(-500, -1, ...)
    
    # 3. Consistency — Invalid categorical
    cat_mask = mask()
    df.loc[cat_mask, "cloud_provider"] = "INVALID_CLOUD"
    
    # 4. Schema — Missing column (file-level)
    if rng.random() < probability:
        df = df.drop(columns=["severity"], errors="ignore")
    
    # 5. Type — Wrong data type
    type_mask = mask()
    df.loc[type_mask, "system_load_before_outage"] = "NOT_A_NUMBER"
    
    # 6. Duplicates
    dup_mask = mask(probability / 2)
    duplicates = df.loc[dup_mask].copy()
    df = pd.concat([df, duplicates], ignore_index=True)
    
    # 7. Outliers
    outlier_mask = ...
    df.loc[outlier_mask, "duration_minutes"] = 9_999_999
```

**Usage:**
```bash
python generate_data_issues.py \
  --input data/cloud_outages_dataset.csv \
  --output data/bad_data/corrupted.csv \
  --probability 0.1
```

**Probability:** 0.1 = 10% of rows affected per error type (configurable).

---

## Model Service (FastAPI)

### `model_service/Dockerfile`

```dockerfile
FROM python:3.12-slim

# Install curl for health checks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY model_service/ .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build context:** Root (`.`) so `COPY requirements.txt` works.  
**Why install curl?** For `healthcheck` in docker-compose.yml.  
**COPY model_service/:** Only copy what's needed (not data/, webapp/, etc.)  
**--host 0.0.0.0:** Bind to all interfaces (required for Docker networking)

---

### `model_service/app/main.py` — API Endpoints

**Lifespan Management**

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    Base.metadata.create_all(bind=engine)  # Create tables
    model, version = load_model()
    if model is None:
        model = train_and_save(DATASET_PATH)  # Train if not found
        version = "v1.0"
    model_state["model"] = model
    model_state["version"] = version
    print(f"Model loaded: {version}")
    yield
    # Shutdown (cleanup if needed)

app = FastAPI(lifespan=lifespan)
```

**Why lifespan?**
- Replaces deprecated `@app.on_event("startup")`
- Runs **before** first request
- Model loaded once → shared across all requests
- Context manager ensures cleanup on shutdown

---

**Endpoint: GET /health**

```python
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model_state["model"] is not None
    }
```

**Purpose:** Docker health check + readiness probe.

---

**Endpoint: POST /predict**

```python
@app.post("/predict", response_model=BatchPredictionResponse)
def make_predictions(request: BatchPredictionRequest, db: Session = Depends(get_db)):
    if model_state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    for feat in request.features:
        hours = predict(model_state["model"], feat.model_dump())
        is_anomaly = hours > 5.0
        
        # Generate random future start time (1-90 days ahead)
        days_ahead = random.randint(1, 90)
        hours_ahead = random.randint(0, 23)
        minutes_ahead = random.randint(0, 59)
        start_time = (datetime.now() + timedelta(
            days=days_ahead, hours=hours_ahead, minutes=minutes_ahead
        )).replace(microsecond=0)
        predicted_end_time = (start_time + timedelta(hours=hours)).replace(microsecond=0)
        
        # Save to DB
        record = Prediction(...)
        db.add(record)
        
        # Build response
        results.append(PredictionResult(...))
    
    db.commit()
    return BatchPredictionResponse(predictions=results)
```

**Key design decisions:**

1. **Batch prediction:** Accepts list of features → single DB commit
2. **Random future dates:** Simulates scheduled predictions (1-90 days out)
3. **Anomaly threshold:** `> 5 hours` flagged as anomaly
4. **Dependency injection:** `Depends(get_db)` provides DB session
5. **Transaction:** Single commit after all predictions

**Why `.replace(microsecond=0)`?**  
Rounds to seconds: `2026-03-17 02:34:35` instead of `2026-03-17 02:34:35.123456`

---

**Endpoint: GET /past-predictions**

```python
@app.get("/past-predictions", response_model=List[PastPrediction])
def past_predictions(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    source: Optional[str] = Query(default="all"),
    limit: int = Query(default=100, le=500),
    db: Session = Depends(get_db),
):
    query = db.query(Prediction)
    if source and source != "all":
        query = query.filter(Prediction.source == source)
    if start_date:
        query = query.filter(Prediction.timestamp >= start_date)
    if end_date:
        query = query.filter(Prediction.timestamp <= end_date)
    return query.order_by(Prediction.timestamp.desc()).limit(limit).all()
```

**Query parameters:**
- `source`: Filter by prediction origin (webapp/scheduled/all)
- `start_date`, `end_date`: Filter by prediction timestamp
- `limit`: Max results (capped at 500 to prevent overload)

**SQLAlchemy query chaining:** Filters added conditionally → single SQL query.

---

### `model_service/app/schemas.py` — Pydantic Models

**Request: PredictionRequest**

```python
class PredictionRequest(BaseModel):
    cloud_provider: str
    service: str
    severity: str
    start_time: Optional[datetime] = None
    system_load_before_outage: Optional[int] = 50
    number_of_customers_affected: int
    ticket_count: int
    backup_system_triggered: str
```

**Why Pydantic?**
- Automatic validation (type checking, required fields)
- Auto-generated OpenAPI docs (`/docs`)
- Serialization/deserialization
- Default values (`system_load_before_outage: Optional[int] = 50`)

---

**Response: PredictionResult**

```python
class PredictionResult(BaseModel):
    # Input features
    cloud_provider: str
    service: str
    severity: str
    start_time: Optional[datetime]
    system_load_before_outage: Optional[int]
    number_of_customers_affected: int
    ticket_count: int
    backup_system_triggered: str
    
    # Model outputs
    predicted_hours: float
    is_anomaly: bool
    model_version: str
    predicted_end_time: Optional[datetime] = None
```

**Echo pattern:** Returns input features + predictions for auditing.

---

**Database Model: PastPrediction**

```python
class PastPrediction(BaseModel):
    id: int
    timestamp: datetime
    model_version: str
    source: str
    # ... all columns from Prediction table
    
    class Config:
        from_attributes = True  # Enable ORM mode
```

**`from_attributes = True`:** Allows creating Pydantic model from SQLAlchemy ORM object.

```python
# SQLAlchemy ORM query
orm_obj = db.query(Prediction).first()

# Convert to Pydantic (thanks to from_attributes=True)
pydantic_obj = PastPrediction.from_orm(orm_obj)
```

---

### `model_service/app/database.py` — SQLAlchemy ORM

**Database Connection**

```python
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
```

**engine:** Connection pool (reuses connections)  
**SessionLocal:** Factory for creating DB sessions  
**Base:** Parent class for ORM models

---

**ORM Model: Prediction**

```python
class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    model_version = Column(String(50), nullable=False)
    source = Column(String(20), default="webapp")
    
    # Input features
    cloud_provider = Column(String(50))
    service = Column(String(50))
    severity = Column(String(20))
    start_time = Column(DateTime(timezone=True), nullable=True)
    system_load_before_outage = Column(Integer)
    number_of_customers_affected = Column(Integer)
    ticket_count = Column(Integer)
    backup_system_triggered = Column(String(5))
    
    # Model outputs
    predicted_hours = Column(Float, nullable=False)
    is_anomaly = Column(Boolean, default=False)
    predicted_end_time = Column(DateTime(timezone=True), nullable=True)
```

**Key points:**
- **`__tablename__`:** Maps to SQL table name
- **DateTime(timezone=True):** Stores timezone-aware timestamps (best practice)
- **default=lambda:** Evaluated at insert time (not class definition)
- **nullable=True:** Optional columns (NULL allowed)

---

**Dependency Injection**

```python
def get_db():
    db = SessionLocal()
    try:
        yield db  # Provide to route handler
    finally:
        db.close()  # Always close (even if exception)
```

**Usage in FastAPI:**

```python
@app.post("/predict")
def make_predictions(request: ..., db: Session = Depends(get_db)):
    # db is a SQLAlchemy session
    db.add(record)
    db.commit()
```

**Benefits:**
- Automatic session lifecycle management
- Connection returned to pool on close
- Exception-safe (finally block)

---

### `model_service/app/model.py` — ML Pipeline

**Feature Engineering**

```python
TIME_FEATURES = ["start_hour", "start_dayofweek", "start_month"]
NUMERIC_FEATURES = [
    "system_load_before_outage",
    "number_of_customers_affected",
    "ticket_count",
    "backup_system_triggered_enc",
] + TIME_FEATURES
CATEGORICAL_FEATURES = ["cloud_provider", "service", "severity"]
```

**Preprocessing Function**

```python
def encode_backup(X):
    X = X.copy()
    
    # Binary encoding for backup_system_triggered
    X["backup_system_triggered_enc"] = X["backup_system_triggered"].map({
        "Yes": 1, "No": 0
    }).fillna(0)
    
    # Extract time features from start_time
    if "start_time" in X.columns:
        start_times = pd.to_datetime(X["start_time"], errors="coerce")
        start_times = start_times.fillna(pd.Timestamp.utcnow())
    else:
        start_times = pd.Series(pd.Timestamp.utcnow(), index=X.index)
    
    X["start_hour"] = start_times.dt.hour           # 0-23
    X["start_dayofweek"] = start_times.dt.dayofweek # 0=Monday, 6=Sunday
    X["start_month"] = start_times.dt.month         # 1-12
    
    return X
```

**Why time features?**  
Outages may have temporal patterns (e.g., more frequent on weekends, during business hours).

**Why fillna(pd.Timestamp.utcnow())?**  
If start_time is missing (old data), use current time as fallback.

---

**Training Pipeline**

```python
def train_and_save(csv_path: str):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[...])  # Remove rows with missing critical fields
    
    X = df[[
        "cloud_provider", "service", "severity", 
        "system_load_before_outage", "number_of_customers_affected",
        "ticket_count", "backup_system_triggered", "start_time"
    ]]
    y = df["duration_minutes"] / 60.0  # Convert minutes → hours
    
    # Preprocessing pipeline
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    
    encoder = FunctionTransformer(encode_backup, validate=False)
    
    # Full pipeline
    pipeline = Pipeline([
        ("encode_backup", encoder),       # Step 1: Feature engineering
        ("preprocessor", preprocessor),   # Step 2: Scale + encode
        ("model", RandomForestRegressor(n_estimators=100, random_state=42)),
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    
    joblib.dump(pipeline, MODEL_PATH)
    return pipeline
```

**Pipeline stages:**
1. **encode_backup:** Custom feature engineering (FunctionTransformer)
2. **preprocessor:** StandardScaler + OneHotEncoder (ColumnTransformer)
3. **model:** RandomForestRegressor

**Why Pipeline?**
- Encapsulates all preprocessing + model
- Single `pipeline.fit()` and `pipeline.predict()`
- Prevents train/test leakage (scaler fitted only on training data)
- Serialization: entire pipeline saved to `.pkl`

**StandardScaler:** Z-score normalization (mean=0, std=1)  
**OneHotEncoder:** Converts categories to binary columns  
**handle_unknown="ignore":** Don't crash on unseen categories (return all zeros)

---

**Inference**

```python
def predict(model, features: dict) -> float:
    df = pd.DataFrame([features])
    return float(model.predict(df)[0])
```

**Input:** Dictionary with feature names  
**Output:** Predicted hours (float)  
**Why DataFrame?** Pipeline expects pandas input (columns must match training).

---

## Webapp (Streamlit)

### `webapp/Dockerfile`

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY webapp/ .

CMD ["streamlit", "run", "1_Prediction.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Entry point:** `1_Prediction.py` (main page)  
**--server.address=0.0.0.0:** Required for Docker (bind to all interfaces)

---

### `webapp/1_Prediction.py` — Prediction Interface

**Page Configuration**

```python
st.set_page_config(page_title="Prediction", layout="wide")
```

**Layouts:**
- `"wide"`: Use full browser width
- `"centered"`: Narrow column (default)

---

**Single Prediction Form**

```python
def build_feature_form(key_suffix=""):
    col1, col2 = st.columns(2)
    with col1:
        cloud_provider = st.selectbox("Cloud Provider", CLOUD_PROVIDERS, key=f"cp_{key_suffix}")
        service = st.selectbox("Service Affected", SERVICES, key=f"svc_{key_suffix}")
        severity = st.selectbox("Incident Severity", SEVERITIES, key=f"sev_{key_suffix}")
    with col2:
        customers = st.number_input("Customers Affected", min_value=0, value=1000, key=f"cust_{key_suffix}")
        tickets = st.number_input("Ticket Count", min_value=0, value=30, key=f"tkt_{key_suffix}")
        backup = st.selectbox("Backup System Triggered", BACKUP_OPTIONS, key=f"bkp_{key_suffix}")
    
    return {
        "cloud_provider": cloud_provider,
        "service": service,
        "severity": severity,
        "number_of_customers_affected": customers,
        "ticket_count": tickets,
        "backup_system_triggered": backup,
    }
```

**st.columns:** Split page into N columns  
**key parameter:** Unique widget ID (prevents conflicts if form reused)  
**Returns dict:** Ready for API request

---

**Making Prediction**

```python
if submitted:
    payload = {"features": [features], "source": "webapp"}
    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        resp.raise_for_status()
        result = resp.json()["predictions"][0]
        
        hours = result["predicted_hours"]
        is_anomaly = result["is_anomaly"]
        start_time = result.get("start_time")
        predicted_end = result.get("predicted_end_time")
        
        st.success(f"Predicted Outage Duration: {hours:.2f} Hours")
        
        # Format datetime for display
        display = {**features, "Predicted Hours": hours, "Anomaly": "Yes" if is_anomaly else "No"}
        if start_time:
            display["Start Time"] = start_time.replace("T", " ").replace("Z", "")
        if predicted_end:
            display["Predicted End Time"] = predicted_end.replace("T", " ").replace("Z", "")
        
        st.dataframe(pd.DataFrame([display]))
        
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Is it running?")
```

**Error handling:**
- `ConnectionError`: API container down
- `raise_for_status()`: HTTP errors (4xx, 5xx)

**Date formatting:**
- API returns: `2026-03-17T02:34:35Z` (ISO 8601)
- Display as: `2026-03-17 02:34:35` (user-friendly)

---

**Batch Prediction (CSV Upload)**

```python
uploaded = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.write("**Preview (first 5 rows):**")
    st.dataframe(df.head())
    
    if st.button("Predict for All Rows"):
        required = ["cloud_provider", "service", "severity",
                    "number_of_customers_affected", "ticket_count", "backup_system_triggered"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.error(f"Missing columns in CSV: {missing}")
        else:
            features_list = df[required].to_dict(orient="records")
            payload = {"features": features_list, "source": "webapp"}
            
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
            preds = resp.json()["predictions"]
            
            results_df = df[required].copy()
            results_df["Predicted Hours"] = [round(p["predicted_hours"], 2) for p in preds]
            results_df["Anomaly"] = ["Yes" if p["is_anomaly"] else "No" for p in preds]
            if "start_time" in preds[0]:
                results_df["Start Time"] = [p.get("start_time", "").replace("T", " ").replace("Z", "") for p in preds]
            if "predicted_end_time" in preds[0]:
                results_df["Predicted End Time"] = [p.get("predicted_end_time", "").replace("T", " ").replace("Z", "") for p in preds]
            
            st.success(f"{len(preds)} predictions made!")
            st.dataframe(results_df)
```

**orient="records":** Converts DataFrame to list of dicts

```python
# DataFrame:
#   col1  col2
# 0   a     1
# 1   b     2

# orient="records":
[{"col1": "a", "col2": 1}, {"col1": "b", "col2": 2}]
```

---

### `webapp/pages/2_Past_Predictions.py` — Historical View

**Multipage App Structure:**
- Main page: `1_Prediction.py`
- Additional pages: `pages/*.py`
- Streamlit auto-discovers pages in `pages/` folder

---

**Filters**

```python
col1, col2, col3 = st.columns(3)
with col1:
    start_date = st.date_input("Start Date", value=date(2024, 1, 1))
with col2:
    end_date = st.date_input("End Date", value=date.today())
with col3:
    source = st.selectbox("Prediction Source", ["all", "webapp", "scheduled"])

limit = st.slider("Max results", 10, 500, 100)
```

---

**Fetching Data**

```python
if st.button("Fetch Predictions"):
    params = {
        "source": source,
        "start_date": datetime.combine(start_date, datetime.min.time()).isoformat(),
        "end_date": datetime.combine(end_date, datetime.max.time()).isoformat(),
        "limit": limit,
    }
    resp = requests.get(f"{API_URL}/past-predictions", params=params, timeout=10)
    data = resp.json()
    
    df = pd.DataFrame(data)
    
    # Format datetime columns
    for col in ["timestamp", "start_time", "predicted_end_time"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace("T", " ").str.rstrip("Z")
    
    st.dataframe(df, use_container_width=True)
```

**Query parameter encoding:**  
`datetime.combine(date, datetime.min.time())` → `2024-01-01 00:00:00`

**.str.rstrip("Z"):** Remove trailing "Z" from ISO timestamps.

---

## Database Layer

### `db/init.sql` — Schema Creation

```sql
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    model_version VARCHAR(50) NOT NULL,
    source VARCHAR(20) DEFAULT 'webapp',
    
    -- Input features
    cloud_provider VARCHAR(50),
    service VARCHAR(50),
    severity VARCHAR(20),
    start_time TIMESTAMPTZ,
    system_load_before_outage INT,
    number_of_customers_affected INT,
    ticket_count INT,
    backup_system_triggered VARCHAR(5),
    
    -- Model outputs
    predicted_hours FLOAT NOT NULL,
    is_anomaly BOOLEAN DEFAULT FALSE,
    predicted_end_time TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS ingestion_stats (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    filename VARCHAR(255),
    total_rows INT,
    valid_rows INT,
    invalid_rows INT,
    criticality VARCHAR(10),
    error_types JSONB
);
```

**SERIAL:** Auto-incrementing integer (PostgreSQL-specific)  
**TIMESTAMPTZ:** Timezone-aware timestamp (stores UTC internally)  
**JSONB:** Binary JSON (for flexible error_types storage)  
**IF NOT EXISTS:** Safe for re-running script

**Why two tables?**
- `predictions`: Operational data (model inference)
- `ingestion_stats`: Monitoring data (data quality tracking)

---

## Data Management

### Dataset Structure

**cloud_outages_dataset.csv** contains historical cloud outage records:

```
incident_id,cloud_provider,region,service,start_time,end_time,duration_minutes,
severity,number_of_customers_affected,ticket_count,backup_system_triggered,...
```

**Key columns:**
- **duration_minutes:** Target variable (convert to hours)
- **start_time, end_time:** Timestamps for time feature extraction
- **Features:** cloud_provider, service, severity, system_load_before_outage, etc.

---

### Folder Strategy

```
data/
├── cloud_outages_dataset.csv   # Original clean data
├── model.pkl                   # Trained model (joblib serialized)
├── raw_data/                   # Ingestion input (batch_*.csv files)
├── good_data/                  # Validated data (passed quality checks)
├── bad_data/                   # Invalid data (failed quality checks)
└── archived_data/              # Post-training data (for future use)
```

**Lifecycle:**
1. **split_dataset.py:** `cloud_outages_dataset.csv` → `raw_data/batch_*.csv`
2. **Ingestion DAG:** `raw_data/` → validate → `good_data/` or `bad_data/`
3. **Prediction DAG:** Read `good_data/` → make predictions
4. **Training DAG:** Read `good_data/` → retrain model → move to `archived_data/`

---

## Docker Infrastructure

### Networking

Docker Compose creates a default bridge network: `dsp-project_default`.

**Service-to-service communication:**
- Webapp → API: `http://api:8000` (DNS name = service name)
- API → DB: `postgresql://postgres:password@db:5432/dsp`

**Host-to-container:**
- Webapp: `http://localhost:8501`
- API: `http://localhost:8000`
- DB: `localhost:5433` (mapped to container's 5432)

---

### Volume Management

**Named volume (postgres_data):**
```yaml
volumes:
  postgres_data:
```

**Storage location:** `/var/lib/docker/volumes/dsp-project_postgres_data`  
**Persists:** Database files survive `docker compose down`  
**Destroyed by:** `docker compose down -v` (removes volumes)

---

**Bind mounts (development):**
```yaml
volumes:
  - ./model_service:/app
  - ./data:/data
```

**Purpose:** Live code reload without rebuilding image.  
**Production:** Remove bind mounts, copy code into image.

---

### Health Checks

**Database health check:**
```yaml
healthcheck:
  test: ["CMD-SHELL", "pg_isready -U postgres"]
  interval: 5s
  timeout: 3s
  retries: 5
```

**pg_isready:** PostgreSQL utility that checks if server is ready to accept connections.

**API health check:**
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 10s
  timeout: 3s
  retries: 3
```

**-f flag:** Fail on HTTP errors (4xx, 5xx).

---

## Machine Learning Pipeline

### Model Architecture

**Random Forest Regressor:**
- Ensemble of 100 decision trees
- Each tree trained on random subset of data (bootstrap)
- Predictions averaged across all trees
- Resistant to overfitting
- Handles non-linear relationships

---

### Feature Engineering

**Original features:**
- cloud_provider (categorical)
- service (categorical)
- severity (categorical)
- system_load_before_outage (numeric)
- number_of_customers_affected (numeric)
- ticket_count (numeric)
- backup_system_triggered (binary string → numeric)

**Derived features (time-based):**
- start_hour (0-23)
- start_dayofweek (0=Mon, 6=Sun)
- start_month (1-12)

**Total features after one-hot encoding:** ~50-70 (depends on unique categories).

---

### Preprocessing Steps

1. **Binary encoding:** "Yes"/"No" → 1/0
2. **Time extraction:** Parse datetime → extract hour/day/month
3. **Scaling:** Numeric features → Z-score normalization
4. **One-hot encoding:** Categorical → binary columns
5. **Model input:** Flattened vector of ~60 features

---

### Model Persistence

**Serialization:** `joblib.dump(pipeline, "model.pkl")`  
**Why joblib?** Efficient for large numpy arrays (used internally by scikit-learn).  
**Includes:** Entire pipeline (preprocessing + model).

**Loading:**
```python
pipeline = joblib.load("model.pkl")
prediction = pipeline.predict(new_data)  # All preprocessing automatic
```

---

## Data Flow & Architecture

### Prediction Flow (Webapp → API → DB)

1. User fills form in Streamlit
2. Streamlit sends POST to `http://api:8000/predict`
3. FastAPI validates request with Pydantic
4. API generates random start_time (1-90 days ahead)
5. API calls `predict()` → scikit-learn pipeline
6. Pipeline preprocesses features → predicts hours
7. API calculates end_time = start_time + hours
8. API saves record to PostgreSQL
9. API returns PredictionResult to Streamlit
10. Streamlit displays result to user

---

### Past Predictions Flow

1. User selects filters (date range, source)
2. Streamlit sends GET to `/past-predictions?start_date=...&source=...`
3. FastAPI queries database with filters
4. SQLAlchemy constructs SQL query
5. PostgreSQL executes query → returns rows
6. FastAPI converts ORM objects → Pydantic models → JSON
7. Streamlit receives JSON → DataFrame → display

---

## API Contracts

### POST /predict

**Request:**
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

**Response:**
```json
{
  "predictions": [
    {
      "cloud_provider": "AWS",
      "service": "Compute",
      "severity": "High",
      "start_time": "2026-04-15T14:23:17",
      "system_load_before_outage": 75,
      "number_of_customers_affected": 1200,
      "ticket_count": 35,
      "backup_system_triggered": "Yes",
      "predicted_hours": 3.42,
      "is_anomaly": false,
      "model_version": "v1.0",
      "predicted_end_time": "2026-04-15T17:48:29"
    }
  ]
}
```

---

### GET /past-predictions

**Query Parameters:**
- `start_date` (optional): ISO datetime
- `end_date` (optional): ISO datetime
- `source` (optional): "webapp" | "scheduled" | "all"
- `limit` (optional, default=100, max=500): Result count

**Example:**
```
GET /past-predictions?source=webapp&start_date=2026-02-01T00:00:00&limit=50
```

**Response:**
```json
[
  {
    "id": 42,
    "timestamp": "2026-02-25T10:15:30Z",
    "model_version": "v1.0",
    "source": "webapp",
    "cloud_provider": "AWS",
    "service": "Compute",
    "severity": "High",
    "start_time": "2026-03-17T02:34:35Z",
    "predicted_hours": 4.48,
    "is_anomaly": false,
    "predicted_end_time": "2026-03-17T07:03:23Z"
  }
]
```

---

## Summary

This project demonstrates a complete ML production system with:

**Architecture:**
- Microservices (FastAPI + Streamlit)
- Containerization (Docker Compose)
- PostgreSQL persistence
- RESTful API design

**ML Pipeline:**
- Feature engineering (time-based + encoding)
- scikit-learn pipeline (preprocessing + model)
- Model persistence (joblib)
- Batch + single prediction support

**Best Practices:**
- Environment variables for configuration
- Health checks for container orchestration
- Dependency injection (FastAPI)
- ORM for database abstraction (SQLAlchemy)
- Pydantic for data validation
- Separation of concerns (API, UI, DB, ML)

**Next Steps (Future Defense 2):**
- Airflow DAGs (ingestion, prediction, training)
- Great Expectations (data validation)
- MLflow (model registry, experiment tracking)
- Grafana (monitoring dashboards)
- CI/CD pipeline (GitHub Actions)
