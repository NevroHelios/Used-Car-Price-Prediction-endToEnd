import pandas as pd
from sklearn.model_selection import train_test_split
from zenml import step
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


@step
def data_splitter(x_train: pd.DataFrame, 
                  test_size: float = 0.2, 
                  random_state: int = 42,
                  sample: bool = True) \
        -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    
    x_train = x_train.drop('clean_title', axis=1) # dropping the clean_title column it has only 1 unique value
    if sample:
        x_train = x_train.sample(frac=0.1, random_state=random_state)
        logging.info(f"Using a sample of the data | Shape: {x_train.shape}")
    
    x_train = x_train.apply(lambda col: col.astype(float) if col.dtype == 'int' else col)

    x = x_train.drop(['price', 'id'], axis=1)
    y = x_train['price']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    logging.info("Data splitter done")
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    a, b, c, d = data_splitter(pd.DataFrame({"hm": [1, 2, 3, 4, 5],
                                             "hm2": [6, 5, 4, 2, 1],
                                             "price": [3, 4, 2, 1, 3]}))

    print(type(a), type(b), type(c), type(d))