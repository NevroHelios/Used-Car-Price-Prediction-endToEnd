import logging
import sqlite3
import time
from pathlib import Path

import pandas as pd


logger = logging.getLogger(__name__)


class SQLiteDataSink:
    def __init__(self, database_path: str, table_name: str) -> None:
        self.database_path = Path(database_path)
        self.table_name = table_name

    def persist(self, df: pd.DataFrame, if_exists: str = "replace") -> str:
        start_time = time.perf_counter()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.database_path) as connection:
            df.to_sql(self.table_name, connection, if_exists=if_exists, index=False)

        duration_seconds = time.perf_counter() - start_time
        logger.info(
            "Persisted dataframe to SQLite: path=%s table=%s rows=%s duration=%.2fs",
            self.database_path,
            self.table_name,
            len(df),
            duration_seconds,
        )
        return str(self.database_path)
