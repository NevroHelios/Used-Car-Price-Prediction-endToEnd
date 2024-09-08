from zenml import step

from src.outlier_detection import (
    DetectOutliers,
    IQROutlierDetection,
    ZScoreOutlierDetection
)
import pandas as pd
import logging


@step
def outlier_detection_step(df: pd.DataFrame, feature: str, strategy: str, method: str = None) -> pd.DataFrame:
    if strategy == 'IQR':
        handler = DetectOutliers(IQROutlierDetection())
    elif strategy == 'ZScore':
        handler = DetectOutliers(ZScoreOutlierDetection())
    else:
        raise ValueError("Method must be either IQR or ZScore")

    abt = handler.handle_outliers(df=df[feature], method=method)
    df_cleaned = df[df[feature].isin(abt[feature])]
    return df_cleaned
