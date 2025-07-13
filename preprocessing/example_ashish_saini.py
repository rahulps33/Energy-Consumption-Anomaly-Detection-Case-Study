import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import signal
from sklearn.preprocessing import MinMaxScaler

def _load_data(file_path, columns):
    # tbd: load the parquet file from file_path and return as pandas DataFrame
    data = pq.read_pandas(file_path, columns = columns).to_pandas()
    return data


def _train_test_split(data, split_rate=0.4):
    # split the data into training and testing sets based on the split_rate parameter
    train_set = data.sample(frac = split_rate, random_state = 25)
    test_set = data.drop(train_set.index)
    return train_set, test_set


def _fill_missing_values(dataset):
    # fill missing values (pay attention to label column)
    filled_dataset = dataset.interpolate(method='linear', limit_direction='both', axis=0)
    return filled_dataset


def _add_artificial_label(dataset):
    # manually label the dataset and append the label as the last column
    dataset['label'] = 0
    labeled_dataset = dataset.copy()
    return labeled_dataset


def _preprocessing(subset):
    # apply multiple preprocessing approaches, e.g. remove trend, normalization
    # detrend the data using signal
    detrend_data = signal.detrend(subset[subset.columns[subset.columns != 'Time']])
    # Normalizing data
    minmax = MinMaxScaler()
    scaled_data = minmax.fit_transform(detrend_data)
    
    preprocessed_dataset = pd.DataFrame(scaled_data, columns = subset.columns.tolist()[1:])
    preprocessed_dataset.insert(0, 'Time', subset['Time'])
    
    return preprocessed_dataset


def get_data(file_path, fill_missing_value=False):
    data = _load_data(file_path)
    labeled_dataset = _add_artificial_label(data)
    if fill_missing_value:
        labeled_dataset = _fill_missing_values(labeled_dataset)
    train_set, test_set = _train_test_split(labeled_dataset, split_rate=0.4)
    preprocessed_train_set, preprocessed_test_set = _preprocessing(train_set), _preprocessing(test_set)
    return preprocessed_train_set.iloc[:, :-1], \
           preprocessed_train_set.iloc[:, -1], \
           preprocessed_test_set.iloc[:, :-1], \
           preprocessed_test_set.iloc[:, -1]
