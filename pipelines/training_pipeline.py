from zenml import pipeline
from zenml.steps import step, Output
from steps.data_ingestion_step import data_ingestion_step
from steps.handling_missing_values_step import handle_missing_values_step


@pipeline(name="used_car_price_predictor")
def ml_pipeline():
    raw_data_artifacts = data_ingestion_step(
        file_path="../data/playground-series-s4e9.zip"
    )
    train, test, sample = raw_data_artifacts

    filled_data = handle_missing_values_step(train)


if __name__ == '__main__':
    ml_pipeline()