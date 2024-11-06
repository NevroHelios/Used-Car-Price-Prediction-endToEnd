import logging
import pandas as pd
from sklearn.pipeline import Pipeline
from zenml import ArtifactConfig, step, Model
from zenml.client import Client
from typing import Annotated, Optional
import mlflow

from sklearn.metrics import  root_mean_squared_error, r2_score

from pipelines.utils import SklearnPipelineMaterializer


# expression_tracker = Client().active_stack.experiment_tracker
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

@step
def evaluation_step(pipeline: Optional[Annotated[Pipeline, ArtifactConfig(materializer=SklearnPipelineMaterializer)]],
                    X_test: pd.DataFrame, y_test: pd.Series) \
                    -> pd.DataFrame:
    
    if pipeline is None or not isinstance(pipeline, Pipeline):
        logging.error(f"Failed to evaluate model: {pipeline} is not a valid pipeline")
        return pd.DataFrame()
    
    # if not mlflow.active_run():
    mlflow.start_run()
    
    try:
        mlflow.sklearn.autolog()
        
        logging.info("Evaluating model")
        y_pred = pipeline.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        
        logging.info("Model evaluation done")
        logging.info(f"Logged Metrices: rmse = {rmse}, r2 = {r2}")
        
        processed_columns = list(pipeline.named_steps['preprocessor'].get_feature_names_out())
        mlflow.log_dict({"processed_columns": processed_columns}, "processed_columns.json")
    
    except Exception as e:
        logging.error("Failed to evaluate model", e)
        raise e
    
    finally:
        mlflow.end_run()
    
    return pd.DataFrame({"rmse": [rmse], "r2": [r2]})