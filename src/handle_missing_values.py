import pandas as pd
from abc import ABC, abstractmethod
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class MissingValuesHandleTemplate(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class DropMissingValuesStrategy(MissingValuesHandleTemplate):
    def __init__(self, axis: int = 0, thresh=None):
        # axis 0 for cols and 1 for rows
        self.axis = axis
        self.thresh = thresh

    def __repr__(self):
        return "DropMissingValuesStrategy"

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f'Dropping missing values with axis = {self.axis}')
        df_cleaned = df.dropna(axis=self.axis, thresh=self.thresh)
        logging.info("Dropped missing values")
        return df_cleaned


class FillMissingValuesStrategy(MissingValuesHandleTemplate):
    def __init__(self, method: str = 'mean', fill_value=None):
        self.method = method
        self.fill_value = fill_value

    def __repr__(self):
        return "FillMissingValuesStrategy"

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        df_cleaned = df.copy()
        if self.method == 'mean':
            num_cols = df_cleaned.select_dtypes('number').columns
            df_cleaned[num_cols] = df_cleaned[num_cols].fillna(df_cleaned[num_cols].mean())
        elif self.method == 'median':
            num_cols = df_cleaned.select_dtypes('number').columns
            df_cleaned[num_cols] = df_cleaned[num_cols].fillna(df_cleaned[num_cols].median())
        elif self.method == 'mode':
            for column in df_cleaned.columns:
                df_cleaned[column].fillna(df_cleaned[column].iloc[0, ], inplace=True)
        elif self.method == "constant":
            df_cleaned = df_cleaned.fillna(self.fill_value)
        else:
            logging.info(f"Invalid method {self.method}")
        logging.info("Missing values filled!")
        return df_cleaned


class MissingValuesHandler:
    def __init__(self, strategy: MissingValuesHandleTemplate = None):
        self._strategy = strategy

    def set_strategy(self, strategy: MissingValuesHandleTemplate):
        self._strategy = strategy
        logging.info(f"Switching missing value handling strategy to {self._strategy}")

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Executing missing value handling strategy: {self._strategy}")
        return self._strategy.handle(df)


if __name__ == '__main__':
    filepath = "../extracted_data/train.csv"
    df = pd.read_csv(filepath)
    strategy = MissingValuesHandler()
    strategy.set_strategy(FillMissingValuesStrategy(method='mode'))
    print(df.isna().sum())
    df = strategy.handle_missing_values(df)
    print(df.isna().sum())
