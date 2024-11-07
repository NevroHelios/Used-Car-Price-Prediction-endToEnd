import logging
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from zenml import ArtifactConfig, step, Model
from zenml.client import Client
import mlflow

from src.feature_engineering import TargetEncode
from sklearn.base import RegressorMixin
# from category_encoders import TargetEncoder

class PipelineRegressor(Pipeline, RegressorMixin):
    """Wrapper for Pipeline to make it a regressor"""
    pass

model=Model(
    name="used_car_price_predictor",
    version="1.0.0",
    description="Used car price predictor",
)
experimental_tracker = Client().active_stack.experiment_tracker
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

@step(enable_cache=False, experiment_tracker=experimental_tracker, model=model)
def model_building_step(x_train: pd.DataFrame, y_train: pd.Series)\
        -> RegressorMixin:
    
    # creating pipelins for numerical and categorical features
    high_cardinality_cols = ["brand", "model", "engine", "ext_col", "int_col", "transmission"]
    low_cardinality_cols = ["fuel_type", "accident"] 

    # define transformers
    preprocessor = ColumnTransformer(
        transformers=[
            # ('group_rare', GroupRareCategories(), high_cardinality_cols),
            ("target_encoder", TargetEncode(), high_cardinality_cols), 
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), low_cardinality_cols),  
        ],
        remainder="passthrough"  
    )

    pipeline = PipelineRegressor(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
        ))
    ])

    if not mlflow.active_run():
        mlflow.start_run()

    try:
        mlflow.sklearn.autolog()

        logging.info(f"Building and training {pipeline.named_steps['regressor']} model")
        pipeline.fit(x_train, y_train)
        logging.info("Model training Completed")

        processed_columns = list(pipeline.named_steps['preprocessor'].get_feature_names_out())

        # Log the columns as a dictionary
        mlflow.log_dict({'used_columns': processed_columns}, 'used_columns.json')

        logging.info(f"Model expects the following columns: {processed_columns}")

    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise e

    finally:
        mlflow.end_run()
    logging.info(f"returning pipeline {pipeline}")
    return pipeline