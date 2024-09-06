import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod


class UniVariateAnalysisTemplate(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature: str, *args, **kwargs) -> None:
        pass


class NumericalUniVariateAnalysis(UniVariateAnalysisTemplate):
    def __init__(self, bins: int = 30):
        self.bins = bins

    def analyze(self, df: pd.DataFrame, feature: str, **kwargs) -> None:
        bins = kwargs.get('bins', self.bins)

        assert feature in df.columns, f"{feature} is not in the dataframe. available features: {df.columns}"

        plt.figure(figsize=(15, 8))
        sns.histplot(df[feature], bins=bins, kde=True)
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.show()


class CategoricalUniVariateAnalysis(UniVariateAnalysisTemplate):
    def analyze(self, df: pd.DataFrame, feature: str, **kwargs) -> None:
        assert feature in df.columns, f"{feature} is not in the dataframe. available features: {df.columns}"

        try:
            plt.figure(figsize=(15, 8))
            sns.countplot(x=feature, data=df, palette="muted")
            plt.title(f"Distribution of {feature}")
            plt.xlabel(feature)
            plt.ylabel("Count")
            plt.xticks(rotation=70)
            plt.show()
        except Exception as e:
            print(f"selected feature is not categorical: {e}")


class UniVariateAnalysis:
    def __init__(self, analyser: UniVariateAnalysisTemplate):
        self.analyser = analyser

    def set_analyser(self, analyser: UniVariateAnalysisTemplate):
        self.analyser = analyser

    def execute_analysis(self, df: pd.DataFrame, feature: str) -> None:
        self.analyser.analyze(df, feature)


if __name__ == '__main__':
    train_df = pd.read_csv('../../extracted_data/train.csv')
    analyser = UniVariateAnalysis(NumericalUniVariateAnalysis())
    analyser.execute_analysis(train_df, feature='milage')
