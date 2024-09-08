from zenml import pipeline
from steps.data_ingestion_step import data_ingestion_step
from steps.handling_missing_values_step import handle_missing_values_step
from steps.feature_engineering_step import feature_engineering_step
from steps.outlier_detection_step import outlier_detection_step
import os
os.environ['NUMEXPR_MAX_THREADS'] = '6'
os.environ['NUMEXPR_NUM_THREADS'] = '2'


@pipeline(name="used_car_price_predictor")
def ml_pipeline():
    raw_data_artifacts = data_ingestion_step(
        file_path="../data/playground-series-s4e9.zip"
    )
    train, test, sample = raw_data_artifacts

    filled_train = handle_missing_values_step(train)
    filled_test = handle_missing_values_step(test)

    engineered_train = feature_engineering_step(filled_train, strategy='log', features=['price'])

    cleaned_train = outlier_detection_step(df=engineered_train, feature='price', strategy='IQR', method='remove')


if __name__ == '__main__':
    ml_pipeline()
