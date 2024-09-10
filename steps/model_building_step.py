import logging

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from zenml import ArtifactConfig, step, Model
from zenml.client import Client
from typing import Annotated
import mlflow

experimental_tracker = Client().active_stack.experiment_tracker
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

model = Model(
    name="Regression Model",
    version="0.1.0",
    license="Apache License 2.0",
    description="Regression Model for predicting used car prices",
)


@step(enable_cache=False, experiment_tracker=experimental_tracker, model=model)
def model_building_step(x_train: pd.DataFrame, y_train: pd.Series)\
        -> Annotated[Pipeline, ArtifactConfig(name="sklearn_pipeline", is_model_artifact=True)]:
    categorical_features = [col for col in x_train.columns if x_train[col].dtype == 'O']
    preprocessor = ColumnTransformer(
        transformers=[
            ('imputer', SimpleImputer(strategy="most_frequent")),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        ))
    ])

    if not mlflow.active_run():
        mlflow.start_run()

    try:
        mlflow.sklearn.autolog()

        logging.info("Building and training XGB model")
        pipeline.fit(x_train, y_train)
        logging.info("Model training Completed")

        processed_columns = list(pipeline.named_steps['preprocessor'].get_feature_names_out(categorical_features)) + \
                            [col for col in x_train.columns if col not in categorical_features]

        # Log the columns as a dictionary
        mlflow.log_dict({'used_columns': processed_columns}, 'used_columns.json')

        logging.info(f"Model expects the following columns: {processed_columns}")

    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise e

    finally:
        mlflow.end_run()

    return pipeline