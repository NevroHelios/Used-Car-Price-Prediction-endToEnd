import os
import pandas as pd
import zipfile
from pathlib import Path
from abc import ABC, abstractmethod


# Define an abstract class for data ingestor
class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, compressed_file_path: str) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """ Ingests the data
        Parameters:
            takes file path as input;
        Returns:
             a pandas dataframe or for kaggle competitions [train, test, sample submission]
        """
        pass


class ZipDataIngestor(DataIngestor):
    def ingest(self, zip_file_path: str) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """ Ingests the data from zip file
        Parameters:
            takes zip file path as input"""
        if not zip_file_path.endswith(".zip"):
            raise Exception("File extension must be .zip")

        # extract the zip data
        root_dir = Path(zip_file_path).parent.parent
        extract_dir = root_dir / "extracted_data"
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        extracted_files = os.listdir(extract_dir)
        csv_files = [f for f in extracted_files if f.endswith(".csv")]

        if len(csv_files) == 0:
            raise Exception("No .csv files found in extracted_data")
        elif len(csv_files) == 1:
            csv_file_path = os.path.join(extract_dir, csv_files[0])
            return pd.read_csv(csv_file_path)
        elif len(csv_files) == 3:
            # if it is from kaggle find the train, test, sample_submission and return them
            csv_file_paths = [os.path.join(extract_dir, f) for f in csv_files]
            train = [f for f in csv_file_paths if f.find("train") != -1][0]
            test = [f for f in csv_file_paths if f.find("test") != -1][0]
            sample = [f for f in csv_file_paths if f.find("sample") != -1][0]
            return pd.read_csv(train), pd.read_csv(test), pd.read_csv(sample)
        else:
            raise Exception("Too many .csv files found in extracted_data")


class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_extension: str) -> DataIngestor:
        """returns the appropriate DataIngestor based on the file extension (currently supports .zip"""
        if file_extension == "zip" or file_extension == ".zip":
            return ZipDataIngestor()
        else:
            raise ValueError(f"No extension found for {file_extension}")


if __name__ == "__main__":
    # get the specific path for the file
    file_path = "E:\kaggle\Regression of Used Car Prices\data\playground-series-s4e9.zip"

    # get the file extension
    file_extension = file_path.split(".")[-1]

    # get the data ingestor
    data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)

    # ingest the data and load it into dataframe
    df_train, df_test, df_sample = data_ingestor.ingest(file_path)

    print(df_train.head())
