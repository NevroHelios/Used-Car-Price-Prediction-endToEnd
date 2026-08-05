# Used Car Price Prediction — End-to-End ML Pipeline

An end-to-end machine learning pipeline for predicting used car prices, built with **ZenML** for orchestration and **MLflow** for experiment tracking. The project uses data from the Kaggle *Playground Series - Season 4, Episode 9* competition.

## Project Structure

```
├── data/                          # Kaggle dataset (playground-series-s4e9.zip)
├── src/                           # Core business logic
│   ├── ingest_data.py             # Data ingestion (Factory pattern)
│   ├── data_quality.py            # Schema, null-rate, and range validation
│   ├── persist_data.py            # Queryable SQLite persistence
│   ├── handle_missing_values.py   # Missing value strategies (drop/fill)
│   ├── feature_engineering.py     # Log transform, scaling, encoding
│   ├── outlier_detection.py       # IQR and Z-Score detection
├── steps/                         # ZenML pipeline steps
│   ├── data_ingestion_step.py     # Extract zip, load CSVs
│   ├── data_quality_validation_step.py
│   ├── handling_missing_values_step.py
│   ├── feature_engineering_step.py
│   ├── outlier_detection_step.py
│   ├── persist_data_step.py       # Persist cleaned data to SQLite
│   ├── data_splitter_step.py      # Train/test split (80/20)
│   ├── model_building_step.py     # RandomForest + preprocessing pipeline
│   ├── evaluation_step.py         # RMSE, R² metrics
├── pipelines/
│   └── training_pipeline.py       # Main pipeline orchestrator
├── analysis/
│   ├── EDA.ipynb                  # Exploratory data analysis notebook
│   └── analytic_src/              # Modular EDA strategies
├── experiments.ipynb              # Feature engineering experiments
├── mlruns/                        # MLflow tracking artifacts
└── requirements.txt
```

## Pipeline

```
Data Ingestion → Data Quality Validation → Handle Missing Values
    → Feature Engineering (log transform) → Outlier Detection (IQR)
    → Persist Cleaned Data → Train/Test Split → Model Building → Evaluation
```

The ingestion and data-quality steps log row counts and step duration for basic pipeline health monitoring.

**Model:** `RandomForestRegressor` with a scikit-learn preprocessing pipeline:
- **High-cardinality features** (brand, model, engine, etc.) — smoothed target encoding
- **Low-cardinality features** (fuel_type, accident) — one-hot encoding

**Metrics:** RMSE and R² logged to MLflow.

## Dataset

- **188,533 samples**, 13 features (4 numerical, 9 categorical)
- Notable: `clean_title` has only 1 unique value and is dropped during training
- Missing values in `fuel_type` (2.6%), `accident` (1.3%), `clean_title` (11.3%)

### Data Sources

The default training run uses the Kaggle zip already stored at `data/playground-series-s4e9.zip`.

The ingestion factory also supports a second concrete implementation, `UrlDataIngestor`, for HTTP(S) CSV, JSON, JSONL, Parquet, and zip sources. A relevant public live-source candidate is the Rebrowser Cars.com Vehicle Listings dataset on Hugging Face, which publishes daily Cars.com listing samples with vehicle specs, mileage, fuel type, dealer location, and inventory condition:

`https://huggingface.co/datasets/rebrowser/carscom-dataset`

That dataset does not match this model's training schema exactly, so it is documented as an alternate ingestion source rather than wired into the default training pipeline.

### Data Quality Gate

`data_quality_validation_step` is a pass/fail validation stage that runs before missing-value and outlier correction. It checks:

- Required schema columns for train and test data
- Numeric vs categorical type expectations
- Null-rate thresholds for nullable fields
- Numeric value ranges for `id`, `model_year`, `milage`, and `price`

The step raises `DataQualityValidationError` on breach and logs validation outcomes.

### Queryable Output

The cleaned, engineered, outlier-filtered training table is persisted to SQLite:

- Database: `data/processed/used_car_features.sqlite`
- Table: `cleaned_used_car_prices`

Example query:

```bash
sqlite3 data/processed/used_car_features.sqlite \
  "SELECT brand, COUNT(*) AS listings, AVG(price) AS avg_log_price
   FROM cleaned_used_car_prices
   GROUP BY brand
   ORDER BY listings DESC
   LIMIT 10;"
```

### Data Dictionary

| Column | Type | Description | Validation |
| --- | --- | --- | --- |
| `id` | Numeric | Unique row identifier | Required, non-null, >= 0 |
| `brand` | Categorical | Vehicle manufacturer | Required, non-null |
| `model` | Categorical | Vehicle model | Required, non-null |
| `model_year` | Numeric | Vehicle model year | Required, 1900-2026 |
| `milage` | Numeric | Odometer mileage | Required, 0-1,000,000 |
| `fuel_type` | Categorical | Fuel category | Nullable, max 5% nulls |
| `engine` | Categorical | Engine description | Required, non-null |
| `transmission` | Categorical | Transmission description | Required, non-null |
| `ext_col` | Categorical | Exterior color | Required, non-null |
| `int_col` | Categorical | Interior color | Required, non-null |
| `accident` | Categorical | Accident/damage report status | Nullable, max 5% nulls |
| `clean_title` | Categorical | Clean title indicator | Nullable, max 15% nulls |
| `price` | Numeric | Target vehicle price in USD | Train only, 1-10,000,000 |

## Design Patterns

The codebase uses **Strategy** and **Factory** patterns throughout for extensibility:
- Interchangeable strategies for missing value handling, feature engineering, and outlier detection
- Factory-based data ingestion supporting local zip files and URL/API-style tabular sources
- Context/handler wrappers providing unified interfaces

## Setup

> Linux or WSL is recommended.

```bash
# Create and activate a virtual environment (Python 3.10–3.11)
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Initialize ZenML
zenml init
zenml up
zenml connect --url=http://127.0.0.1:8237
```

## Usage

```bash
# Run the training pipeline
python pipelines/training_pipeline.py

# Launch MLflow UI to view experiment results
mlflow ui
```

## Dependencies

pandas, numpy, scikit-learn, zenml, mlflow, xgboost, seaborn, matplotlib
