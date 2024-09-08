from typing import List

import pandas as pd
from zenml import step
from src.feature_engineering import (
    FeatureEngineer,
    LogTransform,
    StandardScaling,
    MinMaxScaling,
    OneHotEncoding,
)


@step
def feature_engineering_step(dataframe: pd.DataFrame, strategy: str, features: List[str]) -> pd.DataFrame:
    if strategy == 'log':
        handler = FeatureEngineer(LogTransform(features=features))
    elif strategy == 'standard':
        handler = FeatureEngineer(StandardScaling(features=features))
    elif strategy == 'minmax':
        handler = FeatureEngineer(MinMaxScaling(features=features))
    elif strategy == 'onehotencoding':
        handler = FeatureEngineer(OneHotEncoding(features=features))
    else:
        raise ValueError(f'strategy {strategy} is not supported')

    df_cleaned = handler.apply_feature_engineering(dataframe)
    return df_cleaned


if __name__ == "__main__":
    abt = pd.read_csv("../extracted_data/train.csv")
    abt_cleaned = feature_engineering_step(abt, strategy='log', features=['price'])
    print(abt_cleaned.head())
