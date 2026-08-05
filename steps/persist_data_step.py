import pandas as pd
from zenml import step

from src.persist_data import SQLiteDataSink


DEFAULT_DATABASE_PATH = "data/processed/used_car_pipeline.sqlite"


@step
def persist_bronze_train_step(
    df: pd.DataFrame,
    database_path: str = DEFAULT_DATABASE_PATH,
) -> str:
    sink = SQLiteDataSink(database_path=database_path, table_name="bronze_train")
    return sink.persist(df)


@step
def persist_bronze_test_step(
    df: pd.DataFrame,
    database_path: str = DEFAULT_DATABASE_PATH,
) -> str:
    sink = SQLiteDataSink(database_path=database_path, table_name="bronze_test")
    return sink.persist(df)


@step
def persist_silver_train_step(
    df: pd.DataFrame,
    database_path: str = DEFAULT_DATABASE_PATH,
) -> str:
    sink = SQLiteDataSink(database_path=database_path, table_name="silver_clean_train")
    return sink.persist(df)


@step
def persist_silver_test_step(
    df: pd.DataFrame,
    database_path: str = DEFAULT_DATABASE_PATH,
) -> str:
    sink = SQLiteDataSink(database_path=database_path, table_name="silver_clean_test")
    return sink.persist(df)


@step
def persist_gold_features_step(
    df: pd.DataFrame,
    database_path: str = DEFAULT_DATABASE_PATH,
) -> str:
    sink = SQLiteDataSink(database_path=database_path, table_name="gold_model_features_train")
    return sink.persist(df)


@step
def persist_gold_aggregates_step(
    df: pd.DataFrame,
    database_path: str = DEFAULT_DATABASE_PATH,
) -> str:
    sink = SQLiteDataSink(database_path=database_path, table_name="gold_dashboard_aggregates")
    return sink.persist(df)
