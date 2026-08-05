import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SchemaRule:
    type_family: str
    nullable: bool = False
    max_null_rate: float = 0.0
    min_value: float | None = None
    max_value: float | None = None


USED_CAR_SCHEMA: dict[str, SchemaRule] = {
    "id": SchemaRule("numeric", min_value=0),
    "brand": SchemaRule("categorical"),
    "model": SchemaRule("categorical"),
    "model_year": SchemaRule("numeric", min_value=1900, max_value=2026),
    "milage": SchemaRule("numeric", min_value=0, max_value=1_000_000),
    "fuel_type": SchemaRule("categorical", nullable=True, max_null_rate=0.05),
    "engine": SchemaRule("categorical"),
    "transmission": SchemaRule("categorical"),
    "ext_col": SchemaRule("categorical"),
    "int_col": SchemaRule("categorical"),
    "accident": SchemaRule("categorical", nullable=True, max_null_rate=0.05),
    "clean_title": SchemaRule("categorical", nullable=True, max_null_rate=0.15),
    "price": SchemaRule("numeric", min_value=1, max_value=10_000_000),
}


class DataQualityValidationError(ValueError):
    """Raised when a dataframe breaches a configured data quality rule."""


class DataQualityValidator:
    def __init__(self, schema: dict[str, SchemaRule]) -> None:
        self.schema = schema

    def validate(self, df: pd.DataFrame, target_required: bool = True) -> dict[str, Any]:
        failures: list[str] = []
        columns_to_validate = {
            column: rule
            for column, rule in self.schema.items()
            if target_required or column != "price"
        }

        missing_columns = [column for column in columns_to_validate if column not in df.columns]
        if missing_columns:
            failures.append(f"Missing required columns: {missing_columns}")

        for column, rule in columns_to_validate.items():
            if column not in df.columns:
                continue

            null_rate = float(df[column].isna().mean())
            if not rule.nullable and null_rate > 0:
                failures.append(f"{column} has null_rate={null_rate:.4f}; expected 0")
            elif null_rate > rule.max_null_rate:
                failures.append(
                    f"{column} has null_rate={null_rate:.4f}; threshold={rule.max_null_rate:.4f}"
                )

            if rule.type_family == "numeric" and not is_numeric_dtype(df[column]):
                failures.append(f"{column} must be numeric; found {df[column].dtype}")
                continue
            if rule.type_family == "categorical" and is_numeric_dtype(df[column]):
                failures.append(f"{column} must be categorical; found {df[column].dtype}")
                continue

            if rule.type_family == "numeric":
                non_null_values = df[column].dropna()
                if rule.min_value is not None and (non_null_values < rule.min_value).any():
                    failures.append(f"{column} contains values below {rule.min_value}")
                if rule.max_value is not None and (non_null_values > rule.max_value).any():
                    failures.append(f"{column} contains values above {rule.max_value}")

        report = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "null_rates": df.isna().mean().round(4).to_dict(),
            "failures": failures,
        }

        if failures:
            logger.error("Data quality validation failed: %s", failures)
            raise DataQualityValidationError("; ".join(failures))

        logger.info(
            "Data quality validation passed: rows=%s columns=%s",
            report["row_count"],
            report["column_count"],
        )
        return report
