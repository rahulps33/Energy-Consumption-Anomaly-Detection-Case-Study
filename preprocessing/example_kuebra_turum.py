import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from scipy import signal
from statsmodels.tsa.seasonal import seasonal_decompose

def _load_data(file_path):
    # tbd: load the parquet file from file_path and return as pandas DataFrame
    data = pq.read_table(file_path).to_pandas()
    return data


def _train_test_split(data, split_rate=0.4):
    # split the data into training and testing sets based on the split_rate parameter
    train_set, test_set = train_test_split(data,test_size=split_rate)
    return train_set, test_set


def _fill_missing_values(dataset):
    # fill missing values (pay attention to label column)
    #interpolate
    dataset.iloc[:,1:]  = dataset.iloc[:,1:].interpolate(method ='linear', limit_direction ='both', axis=0)
    return dataset

def _add_artificial_label(dataset):
    # manually label the dataset and append the label as the last column
    dataset['label'] = 0
    labeled_dataset = dataset.copy()    
    return labeled_dataset


def _preprocessing(subset):  
    #apply multiple preprocessing approaches, e.g. remove trend, normalization
    #normalization
    scaler = MinMaxScaler()
    norm_dataset = scaler.fit_transform(subset.iloc[:,1:])
    #detrend
    detrended_df = signal.detrend(norm_dataset)
    #remove seasonality
    res = seasonal_decompose(norm_dataset, model='multiplicative', extrapolate_trend='freq')  
    preprocessed_dataset = pd.DataFrame(detrended_df.values - res.trend)
    #take difference of values 
    preprocessed_dataset= preprocessed_dataset.diff()
    #insert date col 
    preprocessed_dataset.insert(0, 'Date', subset['Date'])
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
