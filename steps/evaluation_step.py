import logging
import pandas as pd
from sklearn.pipeline import Pipeline
from zenml import step
import mlflow
import numpy as np
from sklearn.metrics import  root_mean_squared_error, r2_score
from sklearn.base import RegressorMixin


# expression_tracker = Client().active_stack.experiment_tracker
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

@step
def evaluation_step(model_pipeline: RegressorMixin,
                    X_test: pd.DataFrame, y_test: pd.Series) \
                    -> pd.DataFrame:
    

    rmse, r2 = np.nan, np.nan
    mlflow.start_run()
    
    try:
        mlflow.sklearn.autolog()
        
        logging.info("Evaluating model")
        y_pred = model_pipeline.predict(X_test)

        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        
        logging.info(f"Logged Metrices: rmse = {rmse}, r2 = {r2}")
        
        processed_columns = list(model_pipeline.named_steps['preprocessor'].get_feature_names_out())
        mlflow.log_dict({"processed_columns": processed_columns}, "processed_columns.json")
    
        if hasattr(model_pipeline, 'named_steps'):
            processed_columns = list(model_pipeline.named_steps['preprocessor'].get_feature_names_out())
            mlflow.log_dict({"processed_columns": processed_columns}, "processed_columns.json")

    except Exception as e:
        logging.error("Failed to evaluate model", e)

    finally:
        mlflow.end_run()
    
    return pd.DataFrame({
        "rmse": [rmse],
        "r2": [r2]
    })