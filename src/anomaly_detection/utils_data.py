"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
  
  Licensed under the Apache License, Version 2.0 (the "License").
  You may not use this file except in compliance with the License.
  You may obtain a copy of the License at
  
      http://www.apache.org/licenses/LICENSE-2.0
  
  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
"""

import os
import pickle
from IPython import test
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def get_random_occlusion_mask(dataset, n_intervals, occlusion_prob):
    len_dataset, _, n_features = dataset.shape

    interval_size = int(np.ceil(len_dataset/n_intervals))
    mask = np.ones(dataset.shape)
    
    for i in range(n_intervals):
        u = np.random.rand(n_features)
        mask_interval = (u>occlusion_prob)*1
        mask[i*interval_size:(i+1)*interval_size, :, :] = mask[i*interval_size:(i+1)*interval_size, :, :]*mask_interval

    # Add one random interval for complete missing features 
    feature_sum = mask.sum(axis=0)
    missing_features = np.where(feature_sum==0)[1]
    for feature in missing_features:
        i = np.random.randint(0, n_intervals)
        mask[i*interval_size:(i+1)*interval_size, :, feature] = 1

    random_mask = mask
    missing_values_mask = np.where(pd.isna(dataset), 0, 1)
    #combining random occlusion mask with the missing values mask using AND operation
    combined_mask = np.where(np.logical_and.reduce([random_mask, missing_values_mask]), 1, 0)

    return combined_mask

def dghl_normalization(dataset):
    #dataset = np.ma.array(dataset, mask=np.isnan(dataset))
    normed = (dataset - dataset.mean(axis=0)) / dataset.std(axis=0)
    return normed


def load_buildings(buildings, occlusion_intervals, occlusion_prob, root_dir='./data', verbose=True):
    
    for entity in buildings:
        # ------------------------------------------------- Reading data -------------------------------------------------
        if verbose:
            print(10*'-','entity ', entity, 10*'-')

        train_data = pq.read_pandas(f'{root_dir}/BuildingDataset/train/{entity}.parquet').to_pandas()
        train_data = train_data.values[:, 1:-1]
        train_data = train_data.astype(float)
        train_data = train_data[:, None, :]
        test_data = pq.read_pandas(f'{root_dir}/BuildingDataset/test/{entity}.parquet').to_pandas()
        test_data = test_data.fillna(0)
        test_data = test_data.values[:, 1:-1]
        test_data = test_data.astype(float)
        test_data = test_data[:, None, :]
        len_test = len(test_data)
        test_labels = pq.read_pandas(f'{root_dir}/BuildingDataset/test/{entity}.parquet').to_pandas()
        test_labels = test_labels.values
        test_labels = test_labels[:,-1]
 

        dataset = np.vstack([train_data, test_data])
        if verbose:
            print('Full Train shape: ', train_data.shape)
            print('Full Test shape: ', test_data.shape)
            print('Full Dataset shape: ', dataset.shape)
            print('Full labels shape: ', test_labels.shape)
            print('---------')

    # ------------------------------------------------- Training Occlusion Mask -------------------------------------------------
        # Masks
        mask_filename = f'{root_dir}/BuildingDataset/test/mask_{entity}_{occlusion_intervals}_{occlusion_prob}.parquet'
        if os.path.exists(mask_filename):
            if verbose:
                print(f'Train mask {mask_filename} loaded!')
            
            df = pq.read_pandas(mask_filename).to_pandas()
            train_mask = df.values
            train_mask = train_mask[:, None, :]
        else:
            print('Train mask not found, creating new one')
            print('before_mask_train_data:' , train_data.shape)
            train_mask = get_random_occlusion_mask(dataset=train_data, n_intervals=occlusion_intervals, occlusion_prob=occlusion_prob)
            with open(mask_filename,'wb') as f:
                t_df = pd.DataFrame(np.reshape(train_mask, (train_mask.shape[0], train_mask.shape[-1])))
                table = pa.Table.from_pandas(t_df)
                pq.write_table(table, f)
            if verbose:
                print(f'Train mask {mask_filename} created!')


        # filling the missing values in train_data
        train_data = np.where(np.isnan(train_data), 0, train_data)
        test_mask = np.ones(test_data.shape)

        # calling dghl_normalization
        train_data = dghl_normalization(train_data)
        test_data = dghl_normalization(test_data)

        if verbose:
                print('Train Data shape: ', train_data.shape)
                print('Test Data shape: ', test_data.shape)

                print('Train Mask mean: ', train_mask.mean())
                print('Test Mask mean: ', test_mask.mean())
                print('Train mask shape:', train_mask.shape)

        train_data = [train_data]
        train_mask = [train_mask]
        test_data = [test_data]
        test_mask = [test_mask]
        test_labels = [test_labels]

    return train_data, train_mask, test_data, test_mask, test_labels



