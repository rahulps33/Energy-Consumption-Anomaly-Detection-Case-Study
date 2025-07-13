import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.preprocessing import MinMaxScaler

def lstm_get_data(train_path, test_path):

    # Reading the data
    train = pq.read_pandas(train_path).to_pandas()
    test = pq.read_pandas(test_path).to_pandas()
    train = train.set_index('Time')
    test = test.set_index('Time')

    # Dropping unused columns
    train.drop(['Day', 'Dayofweek', 'Hourofday','Timestamp','Normalize_timestamp'], axis=1, inplace=True)
    test.drop(['Day', 'Dayofweek', 'Hourofday','Timestamp','Normalize_timestamp'], axis=1, inplace=True)

    # Interpolating the missing values
    train.interpolate(inplace=True)
    train.bfill(inplace=True)

    test.interpolate(inplace=True)
    test.bfill(inplace=True)
    
    # Remove columns  that are empty in test from both train and test
    empty_cols = [col for col in test.columns if test[col].isna().all()]
    train.drop(columns=empty_cols, inplace=True)
    test.drop(columns=empty_cols, inplace=True)

    # Scaling the data
    scaler = MinMaxScaler()
    scaler = scaler.fit(train.loc[:, train.columns != 'label'])

    train.loc[:, train.columns != 'label'] = scaler.transform(train.loc[:, train.columns != 'label'])
    test.loc[:, train.columns != 'label'] = scaler.transform(test.loc[:, train.columns != 'label'])

    return train.loc[:, train.columns != 'label'], test

