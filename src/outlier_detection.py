from typing import List
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class OutlierDetectionTemplate(ABC):
    @abstractmethod
    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class ZScoreOutlierDetection(OutlierDetectionTemplate):
    def __init__(self, threshold: float = 3):
        self.threshold = threshold

    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info('Detecting outliers')
        z_scores = np.abs((df - df.mean())/df.std())
        outliers = z_scores > self.threshold
        logging.info(f'Outliers detected with z score thresh hold {self.threshold}')
        return outliers


class IQROutlierDetection(OutlierDetectionTemplate):
    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info('Detecting outliers')
        q1 = df.quantile(0.25)
        q3 = df.quantile(0.75)
        iqr = q3 - q1
        outliers = (df < (q1 - 1.5 * iqr)) | (df > (q3 + 1.5 * iqr))
        logging.info(f"Outliers detected with q1 {q1} and q3 {q3}")
        return outliers


class DetectOutliers:
    def __init__(self, strategy: OutlierDetectionTemplate):
        self.strategy = strategy

    def set_strategy(self, strategy: OutlierDetectionTemplate):
        logging.info(f'Switching Strategy to {strategy}')
        self.strategy = strategy

    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info('Detecting outliers')
        outliers = self.strategy.detect_outliers(df)
        return outliers

    def handle_outliers(self, df: pd.DataFrame, method: str = "remove") -> pd.DataFrame:
        if isinstance(df, pd.Series):
            df = df.to_frame()

        outliers = self.detect_outliers(df)
        if method == "remove":
            logging.info("Removing outliers from the dataset")
            # mask = outliers.astype(bool)
            df_cleaned = df[~outliers.all(axis=1)]
        elif method == "cap":
            logging.info("Capping outliers")
            df_cleaned = df.clip(lower=df.quantile(.01), upper=df.quantile(0.99))
        else:
            logging.warning("Unknown outlier detection method")
            return df

        logging.info("Outliers detecting complete")
        return df_cleaned

    @staticmethod
    def visualize_outliers(df: pd.DataFrame, features: List[str]) -> None:
        logging.info(f"Visualizing outliers for features {features}")
        for feature in features:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=df[feature])
            plt.title(f"Box plot of {feature}")
            plt.show()
        logging.info("Outliers visualizing complete")


if __name__ == "__main__":
    outlier_detector = DetectOutliers(IQROutlierDetection())
    abt = pd.read_csv("../extracted_data/train.csv")
    _abt = outlier_detector.handle_outliers(abt['price'], method="remove")
    print(abt.shape)
    print(_abt.shape)
    abt_cleaned = abt[abt['price'].isin(_abt['price'])]
    print(abt_cleaned.shape)
