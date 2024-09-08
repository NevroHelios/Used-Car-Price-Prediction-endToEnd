import pandas as pd

from src.handle_missing_values import (
    MissingValuesHandler,
    DropMissingValuesStrategy,
    FillMissingValuesStrategy
)
from zenml import step


@step
def handle_missing_values_step(df: pd.DataFrame, strategy: str = 'mode', thresh=None, fill_value=None):
    if strategy == 'drop':
        handler = MissingValuesHandler(DropMissingValuesStrategy(axis=0, thresh=thresh))
    elif strategy in ['mean', 'median', 'mode', 'constant']:
        handler = MissingValuesHandler(FillMissingValuesStrategy(method=strategy, fill_value=fill_value))
    else:
        raise ValueError(f'Strategy {strategy} not recognized')

    cleaned_df = handler.handle_missing_values(df)
    return cleaned_df


if __name__ == '__main__':
    demo_file_path = "../extracted_data/train.csv"
    abt = pd.read_csv(demo_file_path)
    missing_values_handler = MissingValuesHandler(FillMissingValuesStrategy(method='mode'))
    print(abt.isna().sum())
    cleaned_abt = missing_values_handler.handle_missing_values(abt)
    print(cleaned_abt.isna().sum())
