import pandas as pd
from zenml import step

from src.analytics import build_used_car_dashboard_aggregates


@step
def build_gold_aggregates_step(df: pd.DataFrame) -> pd.DataFrame:
    return build_used_car_dashboard_aggregates(df)
