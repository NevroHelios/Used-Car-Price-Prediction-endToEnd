import pandas as pd
from zenml import step

from src.persist_data import SQLiteDataSink


@step
def persist_cleaned_data_step(
    df: pd.DataFrame,
    database_path: str = "data/processed/used_car_features.sqlite",
    table_name: str = "cleaned_used_car_prices",
) -> str:
    sink = SQLiteDataSink(database_path=database_path, table_name=table_name)
    return sink.persist(df)
