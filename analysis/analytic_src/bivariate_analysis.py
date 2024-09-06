import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod


class BivariateAnalysisTemplate(ABC):
    @abstractmethod
    def analysis(self, df: pd.DataFrame, feature1: str, feature2: str, *args, **kwargs) -> None:
        pass

    @staticmethod
    def update_df(df: pd.DataFrame, feature1: str, feature2: str, *args, **kwargs) -> pd.DataFrame:
        min_f1_value = kwargs.get('min_f1_value')
        min_f2_value = kwargs.get('min_f2_value')
        verbose = kwargs.get('verbose')

        abt = df.copy()
        if min_f1_value is not None:
            abt = abt[abt[feature1] > min_f1_value]
            if verbose: print(f"Applied filter for {feature1}")
        if min_f2_value is not None:
            abt = abt[abt[feature2] > min_f2_value]
            if verbose: print(f"Applied filter for {feature2}")

        if verbose: print(f"Original shape: {df.shape}, Filtered shape: {abt.shape}")
        return abt


class NumericalVsNumericalBivariateAnalysis(BivariateAnalysisTemplate):
    def analysis(self, df: pd.DataFrame, feature1: str, feature2: str, *args, **kwargs) -> None:
        plt.figure(figsize=(12, 6))
        plt.figure(figsize=(12, 6))
        sns.scatterplot(x=feature1, y=feature2, data=df)
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.xticks(rotation=90)
        plt.ylabel(feature2)
        plt.show()


class CategoricalVsNumericalBivariateAnalysis(BivariateAnalysisTemplate):
    def analysis(self, df: pd.DataFrame, feature1: str, feature2: str, *args, **kwargs) -> None:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=feature1, y=feature2, data=df)
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.xticks(rotation=80)
        plt.ylabel(feature2)
        plt.show()


class BivariateAnalysis:
    def __init__(self, strategy: BivariateAnalysisTemplate):
        self.strategy = strategy

    def set_analyser(self, strategy: BivariateAnalysisTemplate):
        self.strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, feature1: str, feature2: str, min_f1_value: float = None,
                         min_f2_value: float = None) -> None:
        assert feature1 in df.columns, f"Feature {feature1} must be in {df.columns}"
        assert feature2 in df.columns, f"Feature {feature2} must be in {df.columns}"

        abt = self.strategy.update_df(df, feature1, feature2, min_f1_value=min_f1_value, min_f2_value=min_f2_value)
        self.strategy.analysis(abt, feature1, feature2)


if __name__ == "__main__":
    df = pd.read_csv("../../extracted_data/train.csv")
    bivariate_analysis = BivariateAnalysis(NumericalVsNumericalBivariateAnalysis())
    bivariate_analysis.execute_analysis(df, feature1="model", feature2="price", min_f2_value=2000000)
