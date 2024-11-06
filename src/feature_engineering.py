import pandas as pd
from abc import ABC, abstractmethod
import logging
from typing import List
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class TargetEncode(BaseEstimator, TransformerMixin):
    def __init__(self, categories='auto', k=1, f=1, noise_level=0, random_state=42):
        self.categories = categories
        self.k = k
        self.f = f
        self.noise_level = noise_level
        self.random_state = random_state
        self.encodings = {}
        self.prior = None
        self.feature_names = None
        self.features_names_out_ = None
        
    def add_noise(self, series, noise_level):
        return series * (1 + noise_level * np.random.randn(len(series)))
    
    def fit(self, X, y=None):
        logging.info("target encode fiting")
        if isinstance(X, np.ndarray):
            if self.categories == 'auto':
                self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            else:
                self.feature_names = self.categories
            X = pd.DataFrame(X, columns=self.feature_names)
        
        if self.categories == 'auto':
            self.feature_names = X.columns.tolist()
        elif isinstance(self.categories, str):
            self.feature_names = [self.categories]
        else:
            self.feature_names = self.categories

        self.prior = y.mean()
        
        for feature in self.feature_names:
            temp = pd.DataFrame({'feature': X[feature], 'target': y})
            avg = temp.groupby('feature')['target'].agg(['count', 'mean'])
            smoothing = 1 / (1 + np.exp(-(avg['count'] - self.k) / self.f))
            encoded_values = self.prior * (1 - smoothing) + avg['mean'] * smoothing
 
            self.encodings[feature] = encoded_values.to_dict()
        logging.info("target encode fiting complete")
        return self
    
    def transform(self, X):
        logging.info("target encode transform")
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)
            
        Xt = X.copy()
        
        for feature in self.feature_names:
            Xt[feature] = Xt[feature].map(self.encodings[feature])
            Xt[feature] = Xt[feature].fillna(self.prior)
            
            if self.noise_level > 0:
                if self.random_state is not None:
                    np.random.seed(self.random_state)
                Xt[feature] = self.add_noise(Xt[feature], self.noise_level)
        logging.info("target encode transform complete")
        return Xt.values
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
    
    # @staticmethod
    def get_feature_names_out(self, features_names=None):
        logging.info("getting feature names")
        if features_names is not None:
            return np.asarray(features_names, dtype=str)
        return self.feature_names


class GroupRareCategories(BaseEstimator, TransformerMixin):
    def __init__(self, categories='auto', threshold='mean'):
        self.categories = categories
        self.threshold = threshold
        self.rare_categories = {}
        self.feature_names_out_ = []
        self.feature_names = None

    def fit(self, X, y=None):
        logging.info("GroupRareCategories: Starting fit")

        if isinstance(X, np.ndarray):
            if self.categories == 'auto':
                self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            else:
                self.feature_names = self.categories
            X = pd.DataFrame(X, columns=self.feature_names)
        else:
            self.feature_names = X.columns.tolist() if self.categories == 'auto' else self.categories

        for col in self.feature_names:
            freq_count = X[col].value_counts()
            threshold_value = freq_count.mean() if self.threshold == 'mean' else self.threshold
            self.rare_categories[col] = freq_count[freq_count < threshold_value].index
            self.feature_names_out_.append(col)

        logging.info("GroupRareCategories: Fit complete")
        return self

    def transform(self, X):
        logging.info("GroupRareCategories: Starting transform")

        # Ensure X is a DataFrame for consistency
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)

        Xt = X.copy()
        for col, rare_categories in self.rare_categories.items():
            Xt[col] = Xt[col].apply(lambda x: f"other_{col}" if x in rare_categories else x)

        logging.info("GroupRareCategories: Transform complete")
        return Xt.values

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, feature_names=None):
        logging.info("GroupRareCategories: Getting feature names")
        return np.array(self.feature_names_out_, dtype=str)


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
    def __init__(self, features: List[str], cardinality_threshold: int = 7):
        self.features = features
        self.ohe = OneHotEncoder()
        self.cardinality_threshold = cardinality_threshold

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
