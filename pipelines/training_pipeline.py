from zenml import pipeline, Model, step

from steps.data_ingestion_step import data_ingestion_step

@pipeline(
    name="used_car_price_predictor"
)
def ml_pipeline():
    raw_data = data_ingestion_step(
        file_path="../data/playground-series-s4e9.zip"
    )
