import pandas as pd
from src.ingest_data import DataIngestorFactory
from zenml import step
from typing import Tuple


@step
def data_ingestion_step(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    file_extension = '.zip'
    data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)
    _train, _test, _sample = data_ingestor.ingest(file_path)  # it's a kaggle competition data
    return _train, _test, _sample


if __name__ == '__main__':
    demo_file_path = "../data/playground-series-s4e9.zip"
    df = data_ingestion_step(demo_file_path)
    train, test, sample = df
    print(train.shape, test.shape, sample.shape)