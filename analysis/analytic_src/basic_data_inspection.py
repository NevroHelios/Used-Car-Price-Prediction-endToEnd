import pandas as pd
from abc import ABC, abstractmethod


class DataInspectionStrategy(ABC):
    """
    only to be inherited by other classes
    """
    @abstractmethod
    def inspect(self, df: pd.DataFrame) -> None:
        """
        performs a specific type of data inspection
        :parameter:
            df (pd.DataFrame) : on which you want to inspect
        :return:
            None: this method prints the inspection results
        """
        pass


class DataTypesInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame) -> None:
        """
        prints the data types and non-null counts of each column in Dataset
        :parameter:
        df (pd.DataFrame) : on which you want to inspect
        :return:
        None: this method prints the inspection results
        """
        print("\nData types and Non-Null Count")
        print(df.info())
        print("\n")
        print("Null Count")
        print(df.isnull().sum())


class SummaryInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame) -> None:
        """
        prints the summary statistics of each column in Dataset
        :param df:
        :return:
        None: this method prints the inspection results
        """
        print("\nSummary Statistics")
        print("\n")
        print(df.describe())
        print("\n")
        print(df.describe(include=["O"]))


class DataInspector:
    """
    This class allows you to inspect datasets based on a specific strategy.
    """
    def __init__(self, strategy: DataInspectionStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: DataInspectionStrategy):
        self._strategy = strategy

    def execute_inspection(self, df: pd.DataFrame):
        self._strategy.inspect(df)


if __name__ == "__main__":
    csv_path = "../../extracted_data/train.csv"
    df = pd.read_csv(csv_path)
    inspector = DataInspector(DataTypesInspectionStrategy())
    # inspector.set_strategy(SummaryInspectionStrategy())
    inspector.execute_inspection(df)