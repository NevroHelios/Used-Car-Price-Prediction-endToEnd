from zenml import pipeline, Model, get_pipeline_context
import numpy as np
import pandas as pd
import os
import sys
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['NUMEXPR_MAX_THREADS'] = '6'
os.environ['NUMEXPR_NUM_THREADS'] = '2'

from steps.data_ingestion_step import data_ingestion_step
from steps.handling_missing_values_step import handle_missing_values_step
from steps.feature_engineering_step import feature_engineering_step
from steps.outlier_detection_step import outlier_detection_step
from steps.data_splitter_step import data_splitter
from steps.model_building_step import model_building_step
from steps.evaluation_step import evaluation_step

@pipeline(name="used_car_price_predictor",)
def ml_pipeline():
    raw_data_artifacts = data_ingestion_step(
        file_path="data/playground-series-s4e9.zip"
    )
    train, test, sample = raw_data_artifacts

    filled_train = handle_missing_values_step(train)
    filled_test = handle_missing_values_step(test)

    engineered_train = feature_engineering_step(filled_train, strategy='log', features=['price'])
    

    cleaned_train = outlier_detection_step(df=engineered_train, feature='price', strategy='IQR', method='remove')

    X_train, X_test, y_train, y_test = data_splitter(cleaned_train)

    model_pipeline = model_building_step(X_train, y_train)
     
    metrices = evaluation_step(model_pipeline=model_pipeline, X_test=X_test, y_test=y_test)
    


if __name__ == '__main__':
    ml_pipeline()
