# Used Car Price Prediction — End-to-End ML Pipeline

An end-to-end machine learning pipeline for predicting used car prices, built with **ZenML** for orchestration and **MLflow** for experiment tracking. The project uses data from the Kaggle *Playground Series - Season 4, Episode 9* competition.

## Project Structure

```
├── data/                          # Kaggle dataset (playground-series-s4e9.zip)
├── src/                           # Core business logic
│   ├── ingest_data.py             # Data ingestion (Factory pattern)
│   ├── handle_missing_values.py   # Missing value strategies (drop/fill)
│   ├── feature_engineering.py     # Log transform, scaling, encoding
│   ├── outlier_detection.py       # IQR and Z-Score detection
├── steps/                         # ZenML pipeline steps
│   ├── data_ingestion_step.py     # Extract zip, load CSVs
│   ├── handling_missing_values_step.py
│   ├── feature_engineering_step.py
│   ├── outlier_detection_step.py
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
Data Ingestion → Handle Missing Values → Feature Engineering (log transform)
    → Outlier Detection (IQR) → Train/Test Split → Model Building → Evaluation
```

**Model:** `RandomForestRegressor` with a scikit-learn preprocessing pipeline:
- **High-cardinality features** (brand, model, engine, etc.) — smoothed target encoding
- **Low-cardinality features** (fuel_type, accident) — one-hot encoding

**Metrics:** RMSE and R² logged to MLflow.

## Dataset

- **188,533 samples**, 13 features (4 numerical, 9 categorical)
- Notable: `clean_title` has only 1 unique value and is dropped during training
- Missing values in `fuel_type` (2.6%), `accident` (1.3%), `clean_title` (11.3%)

## Design Patterns

The codebase uses **Strategy** and **Factory** patterns throughout for extensibility:
- Interchangeable strategies for missing value handling, feature engineering, and outlier detection
- Factory-based data ingestion supporting multiple file formats
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
