import pandas as pd


def build_used_car_dashboard_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    aggregate_source = df.copy()
    aggregate_source["model_year_bucket"] = pd.cut(
        aggregate_source["model_year"],
        bins=[0, 1999, 2009, 2019, 2026],
        labels=["pre_2000", "2000_2009", "2010_2019", "2020_2026"],
        include_lowest=True,
    )

    aggregates = (
        aggregate_source.groupby(["brand", "model_year_bucket"], observed=True)
        .agg(
            row_count=("id", "count"),
            avg_price=("price", "mean"),
            median_price=("price", "median"),
            avg_milage=("milage", "mean"),
            min_model_year=("model_year", "min"),
            max_model_year=("model_year", "max"),
        )
        .reset_index()
    )
    aggregates["run_timestamp"] = pd.Timestamp.utcnow().isoformat()
    return aggregates
