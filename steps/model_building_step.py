import logging

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from zenml import ArtifactConfig, step, Model
from zenml.client import Client
from typing import Annotated
import mlflow

from src.feature_engineering import TargetEncode, GroupRareCategories
from pipelines.utils import SklearnPipelineMaterializer
# from category_encoders import TargetEncoder

experimental_tracker = Client().active_stack.experiment_tracker
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

model = Model(
    name="Regression Model",
    version="0.1.0", 
    license="Apache License 2.0",
    description="Regression Model for predicting used car prices",
)


@step(enable_cache=True, experiment_tracker=experimental_tracker, model=model)
def model_building_step(x_train: pd.DataFrame, y_train: pd.Series)\
        -> Annotated[Pipeline, ArtifactConfig(materializer=SklearnPipelineMaterializer, is_model_artifact=True)]:
    
    # creating pipelins for numerical and categorical features
    high_cardinality_cols = ["brand", "model", "engine", "ext_col", "int_col", "transmission"]
    low_cardinality_cols = ["fuel_type", "accident"] 
    binary_cols = ["accident"]
    # binary_lookup = {s: i for }

    # def group_rare_categories(df, col, threshold):
    #     freq_counts = df[col].value_counts()
    #     rare_categories = freq_counts[freq_counts < threshold].index
    #     df[col] = df[col].apply(lambda x: f"other_{col}" if x in rare_categories else x)
    #     return df

    # for col in high_cardinality_cols:
    #     group_rare_categories(x_train, col, threshold=x_train[col].value_counts().mean()) 

    # define transformers
    preprocessor = ColumnTransformer(
        transformers=[
            # ('group_rare', GroupRareCategories(), high_cardinality_cols),
            ("target_encoder", TargetEncode(), high_cardinality_cols), 
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), low_cardinality_cols),  
            # ("binary", FunctionTransformer(lambda x: x), binary_cols),  
        ],
        remainder="passthrough"  
    )

    # pipeline = Pipeline(steps=[
    #     ('preprocessor', preprocessor),
    #     ('regressor', XGBRegressor(
    #         objective='reg:squarederror',
    #         n_estimators=100,
    #         learning_rate=0.1,
    #         max_depth=3,
    #         subsample=0.8,
    #         colsample_bytree=0.8,
    #         random_state=42,
    #         tree_method="hist", 
    #         device='cuda',
    #         verbosity=2
    #     ))
    # ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            warm_start=True,
            n_estimators=100
        ))
    ])

    if not mlflow.active_run():
        mlflow.start_run()

    try:
        mlflow.sklearn.autolog()

        logging.info("Building and training RF model")
        pipeline.fit(x_train, y_train)
        logging.info("Model training Completed")

        # processed_columns = list(pipeline.named_steps['preprocessor'].get_feature_names_out(categorical_features)) + \
        #                     [col for col in x_train.columns if col not in categorical_features]

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