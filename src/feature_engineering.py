import pandas as pd
from abc import ABC, abstractmethod
import logging
from typing import List
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class FeatureEngineeringTemplate(ABC):
    @abstractmethod
    def apply_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class LogTransform(FeatureEngineeringTemplate):
    def __init__(self, features: List[str]):
        self.features = features

    def apply_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Applying Log Transform")
        df_transformed = df.copy()
        for feature in self.features:
            df_transformed[feature] = np.log1p(
                df[feature]
            )
        logging.info("Log Transform Done")
        return df_transformed


class StandardScaling(FeatureEngineeringTemplate):
    def __init__(self, features: List[str]):
        self.features = features
        self.scaler = StandardScaler()

    def apply_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Applying Standard Scaling")
        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.fit_transform(df[self.features])
        logging.info("Standard Scaling Done")
        return df_transformed


class MinMaxScaling(FeatureEngineeringTemplate):
    def __init__(self, features: List[str]):
        self.features = features
        self.scaler = MinMaxScaler()

    def apply_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Applying MinMax Scaling")
        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.fit_transform(df[self.features])
        logging.info("MinMax Scaling Done")
        return df_transformed


class OneHotEncoding(FeatureEngineeringTemplate):
    def __init__(self, features: List[str]):
        self.features = features
        self.ohe = OneHotEncoder()

    def apply_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Applying OneHot Encoding")
        df_transformed = df.copy()
        encoded_df = pd.DataFrame(
            self.ohe.fit_transform(df[self.features]),
            columns=self.ohe.get_feature_names_out(self.features)
        )
        df_transformed = df_transformed.drop(columns=self.features)
        df_transformed = pd.concat([df_transformed, encoded_df])
        logging.info("OneHot Encoding Done")
        return df_transformed


class FeatureEngineer:
    def __init__(self, strategy: FeatureEngineeringTemplate):
        self._strategy = strategy

    def set_strategy(self, strategy: FeatureEngineeringTemplate):
        self._strategy = strategy

    def apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Applying Feature Engineering")
        return self._strategy.apply_transform(df)


if __name__ == "__main__":
    abt = pd.read_csv("../extracted_data/train.csv")
    log_transformer = FeatureEngineer(LogTransform(['price']))
    df_cleaned = log_transformer.apply_feature_engineering(abt)
    print(df_cleaned.head())
