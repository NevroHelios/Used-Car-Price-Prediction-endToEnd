import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod


class MultiVariateTemplate(ABC):
    def analyze(self, df: pd.DataFrame) -> None:
        """
        Analyze the given DataFrame.

        This function generates a pairplot for all columns in the DataFrame,
        and a correlation heatmap for all numerical columns.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to analyze.

        Returns
        -------
        None
        """

        num_cols = [col for col in df.columns if df[col].dtype != 'O']
        self.generate_corr_heatmap(df[num_cols])
        self.generate_pairplot(df)

    @staticmethod
    def generate_corr_heatmap(df: pd.DataFrame) -> None:
        pass

    @staticmethod
    def generate_pairplot(df: pd.DataFrame) -> None:
        pass


class SimpleMultiVariateAnalysis(MultiVariateTemplate):
    def generate_corr_heatmap(self, df: pd.DataFrame) -> None:
        """
        Generate a correlation heatmap for the given DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to generate the heatmap for.

        Returns
        -------
        None
        """

        plt.figure(figsize=(12, 10))
        sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.show()

    def generate_pairplot(self, df: pd.DataFrame) -> None:
        """
        Generate a pairplot for the given DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to generate the pairplot for.

        Returns
        -------
        None
        """
        plt.title("Pair plot for selected features")
        sns.pairplot(df)
        plt.show()


class MultiVariateAnalysis:
    def __init__(self, analyser: MultiVariateTemplate) -> None:
        self.analyser = analyser

    def execute_analysis(self, df: pd.DataFrame) -> None:
        """
        Execute the analysis strategy for the given DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to analyze.

        Returns
        -------
        None
        """

        self.analyser.analyze(df)

    def set_analyser(self, strategy: MultiVariateTemplate) -> None:
        """
        Set the analysis strategy for this MultiVariateAnalysis object.

        Parameters
        ----------
        strategy : MultiVariateTemplate
            The strategy to use for the analysis.

        Returns
        -------
        None
        """
        self.analyser = strategy


if __name__ == '__main__':
    df = pd.read_csv('../../extracted_data/train.csv')
    multivariate_analysis = MultiVariateAnalysis(SimpleMultiVariateAnalysis())
    multivariate_analysis.execute_analysis(df)