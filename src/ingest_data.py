import io
import os
import pandas as pd
import zipfile
from pathlib import Path
from abc import ABC, abstractmethod
from urllib.parse import urlparse
from urllib.request import urlopen


# Define an abstract class for data ingestor
class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, source: str) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """ Ingests the data
        Parameters:
            takes source path or URL as input;
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

        parsed_path = urlparse(zip_file_path)
        if parsed_path.scheme in {"http", "https"}:
            with urlopen(zip_file_path) as response:
                zip_bytes = io.BytesIO(response.read())
            return self._read_zip_contents(zip_bytes)

        # extract the zip data
        root_dir = Path(zip_file_path).parent.parent
        extract_dir = root_dir / "extracted_data"
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        extracted_files = os.listdir(extract_dir)
        csv_files = [str(extract_dir / f) for f in extracted_files if f.endswith(".csv")]

        return self._read_csv_files(csv_files)

    def _read_zip_contents(self, zip_source) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        with zipfile.ZipFile(zip_source, "r") as zip_ref:
            csv_files = [file_name for file_name in zip_ref.namelist() if file_name.endswith(".csv")]

            if len(csv_files) == 0:
                raise Exception("No .csv files found in zip source")
            elif len(csv_files) == 1:
                with zip_ref.open(csv_files[0]) as csv_file:
                    return pd.read_csv(csv_file)
            elif len(csv_files) == 3:
                train = [file_name for file_name in csv_files if "train" in file_name][0]
                test = [file_name for file_name in csv_files if "test" in file_name][0]
                sample = [file_name for file_name in csv_files if "sample" in file_name][0]
                with zip_ref.open(train) as train_file, zip_ref.open(test) as test_file, zip_ref.open(sample) as sample_file:
                    return pd.read_csv(train_file), pd.read_csv(test_file), pd.read_csv(sample_file)
            else:
                raise Exception("Too many .csv files found in zip source")

    def _read_csv_files(self, csv_files: list[str]) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if len(csv_files) == 0:
            raise Exception("No .csv files found in extracted_data")
        elif len(csv_files) == 1:
            return pd.read_csv(csv_files[0])
        elif len(csv_files) == 3:
            # if it is from kaggle find the train, test, sample_submission and return them
            train = [f for f in csv_files if f.find("train") != -1][0]
            test = [f for f in csv_files if f.find("test") != -1][0]
            sample = [f for f in csv_files if f.find("sample") != -1][0]
            return pd.read_csv(train), pd.read_csv(test), pd.read_csv(sample)
        else:
            raise Exception("Too many .csv files found in extracted_data")


class UrlDataIngestor(DataIngestor):
    def ingest(self, source_url: str) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Ingests tabular data from an HTTP(S) source."""
        parsed_url = urlparse(source_url)
        if parsed_url.scheme not in {"http", "https"}:
            raise ValueError("UrlDataIngestor requires an HTTP(S) URL")

        clean_path = parsed_url.path.lower()
        if clean_path.endswith(".zip"):
            return ZipDataIngestor().ingest(source_url)
        if clean_path.endswith(".csv"):
            return pd.read_csv(source_url)
        if clean_path.endswith(".json"):
            return pd.read_json(source_url)
        if clean_path.endswith(".jsonl"):
            return pd.read_json(source_url, lines=True)
        if clean_path.endswith(".parquet"):
            return pd.read_parquet(source_url)

        raise ValueError(
            "Unsupported URL data format. Expected .csv, .json, .jsonl, .parquet, or .zip"
        )


class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(source_type: str) -> DataIngestor:
        """Returns the appropriate DataIngestor based on the file extension or source type."""
        normalized_source_type = source_type.lower().lstrip(".")
        if normalized_source_type == "zip":
            return ZipDataIngestor()
        elif normalized_source_type in {"url", "api", "http", "https"}:
            return UrlDataIngestor()
        else:
            raise ValueError(f"No data ingestor found for {source_type}")


if __name__ == "__main__":
    file_path_demo = Path(__file__).resolve().parents[1] / "data" / "playground-series-s4e9.zip"

    file_extension = file_path_demo.suffix
    data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)
    df_train, df_test, df_sample = data_ingestor.ingest(str(file_path_demo))

    print(df_train.head())
