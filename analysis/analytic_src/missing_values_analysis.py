import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod


# abstract base class for missing value analysis
class MissingValuesAnalysisTemplate(ABC):
    def analysis(self, df: pd.DataFrame) -> None:
        self.identify_missing_values(df)
        self.visualize_missing_values(df)

    @abstractmethod
    def identify_missing_values(self, df: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def visualize_missing_values(self, df: pd.DataFrame) -> None:
        pass


class SimpleMissingValuesAnalysis(MissingValuesAnalysisTemplate):
    def identify_missing_values(self, df: pd.DataFrame) -> None:
        missing_values_count = df.isnull().sum()
        print("Missing Values")
        print(missing_values_count[missing_values_count > 0])

    def visualize_missing_values(self, df: pd.DataFrame) -> None:
        print("\nvisualizing missing values")
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
        plt.show()


if __name__ == "__main__":
    df = pd.read_csv('../../extracted_data/train.csv')
    SimpleMissingValuesAnalysis().analysis(df)
