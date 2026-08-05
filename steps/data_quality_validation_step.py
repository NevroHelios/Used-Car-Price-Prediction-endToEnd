import logging
import time
from typing import Tuple

import pandas as pd
from zenml import step

from src.data_quality import DataQualityValidator, USED_CAR_SCHEMA


logger = logging.getLogger(__name__)


@step
def data_quality_validation_step(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    start_time = time.perf_counter()
    validator = DataQualityValidator(USED_CAR_SCHEMA)

    train_report = validator.validate(train, target_required=True)
    test_report = validator.validate(test, target_required=False)

    duration_seconds = time.perf_counter() - start_time
    logger.info(
        "Data quality step completed in %.2fs: train_rows=%s test_rows=%s",
        duration_seconds,
        train_report["row_count"],
        test_report["row_count"],
    )
    return train, test
