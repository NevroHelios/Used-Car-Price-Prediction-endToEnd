import logging
import time
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from src.ingest_data import DataIngestorFactory
from zenml import step
from typing import Tuple


logger = logging.getLogger(__name__)


@step
def data_ingestion_step(
    file_path: str,
    source_type: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    start_time = time.perf_counter()
    parsed_path = urlparse(file_path)
    inferred_source_type = source_type
    if inferred_source_type is None:
        inferred_source_type = parsed_path.scheme if parsed_path.scheme in {"http", "https"} else Path(file_path).suffix

    data_ingestor = DataIngestorFactory.get_data_ingestor(inferred_source_type)
    _train, _test, _sample = data_ingestor.ingest(file_path)  # it's a kaggle competition data

    duration_seconds = time.perf_counter() - start_time
    logger.info(
        "Ingested data from %s using %s in %.2fs: train_rows=%s test_rows=%s sample_rows=%s",
        file_path,
        data_ingestor.__class__.__name__,
        duration_seconds,
        len(_train),
        len(_test),
        len(_sample),
    )
    return _train, _test, _sample


if __name__ == '__main__':
    demo_file_path = "../data/playground-series-s4e9.zip"
    df = data_ingestion_step(demo_file_path)
    train, test, sample = df
    print(train.shape, test.shape, sample.shape)
