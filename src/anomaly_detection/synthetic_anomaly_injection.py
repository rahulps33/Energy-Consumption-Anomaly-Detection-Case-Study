import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import random
import warnings
warnings.filterwarnings('ignore')

# masking function
def get_random_occlusion_mask(dataset, n_intervals, occlusion_prob):
    len_dataset, _, n_features = dataset.shape

    interval_size = int(np.ceil(len_dataset / n_intervals))
    mask = np.ones(dataset.shape)

    for i in range(n_intervals):
        u = np.random.rand(n_features)
        mask_interval = (u > occlusion_prob) * 1
        mask[i * interval_size:(i + 1) * interval_size, :, :] = mask[i * interval_size:(i + 1) * interval_size, :,
                                                                :] * mask_interval

    # Add one random interval for complete missing features
    feature_sum = mask.sum(axis=0)
    missing_features = np.where(feature_sum == 0)[1]
    for feature in missing_features:
        i = np.random.randint(0, n_intervals)
        mask[i * interval_size:(i + 1) * interval_size, :, feature] = 1

    random_mask = mask
    missing_values_mask = np.where(pd.isna(dataset), 0, 1)
    # combining random occlusion mask with the missing values mask using AND operation
    combined_mask = np.where(np.logical_and.reduce([random_mask, missing_values_mask]), 1, 0)

    return combined_mask


# normalization
def dghl_normalization(dataset):
    dataset = np.ma.array(dataset, mask=np.isnan(dataset))
    normed = (dataset - dataset.mean(axis=0)) / dataset.std(axis=0)
    return normed.data


def normalize(df):
    train_data = df.values[:, 1:-1]
    train_data

    train_mask = get_random_occlusion_mask(dataset=train_data[:, None, :], n_intervals=1, occlusion_prob=0)
    train_mask = train_mask[:, 0, :]

    train_data = train_data * train_mask
    train_data = train_data.astype(float)

    arr = dghl_normalization(dataset=train_data)
    df.iloc[:, 1:-1] = arr

    df_feature = df.copy()
    for col in df.columns:
        df_feature[col] = df['label'].values

    return df, df_feature


# synthetic anomaly generation
def synthetic_anomaly(df):
    step_size = 1440
    size = df.shape[0]

    # source dimension
    data = []
    df_source = pd.DataFrame(data)
    for i in range(0, size, step_size):
        df_source = df_source.append(df[df['label'] == 0].sample(n=5, random_state=i, replace=False))
    df_source = df_source[source_col].copy()
    df_source = df_source.sample(1, axis=1)
    df_source['label'] = 1
    df.loc[df_source.index, source_col] = df_source

    # target dimension
    data = []
    df_target = pd.DataFrame(data)
    shape = df_source.shape[0]
    df1 = df[df['label'] == 0]
    nrows = range(df1.shape[0])
    ix = random.randint(nrows.start, nrows.stop - shape)
    df_target = df1.iloc[ix:ix + shape, :]
    df_target = df_target[target_col].copy()
    df_target = df_target.sample(1, axis=1)
    df_target['label'] = 1
    df.loc[df_target.index, target_col] = df_target

    return df_source, df_target, df


# function to generate contextual anomaly
def contextual_anomaly(source, target, df):
    data = []
    df_swapped = pd.DataFrame(data)

    if (len(source) == len(target)):
        df_swapped = df.loc[target.index]
        for i in range(0, 1):
            df_swapped[target.columns[i]] = source[source.columns[i]].values

        df_swapped['label'] = 1

        df.loc[df_swapped.index, target_col] = df_swapped

        df.loc[df_swapped.index, 'label'] = 1
        df.loc[df_swapped.index, 'synthetic_label'] = 1

    return df, df_swapped

# function to generate point anomaly
def point_outlier(source, target, df):
    spike_loc = 200
    data = []
    df_point = pd.DataFrame(data)

    df1 = df[df['label'] == 0]

    nrows = range(df1.shape[0])
    ix = random.randint(nrows.start, nrows.stop - spike_loc)
    df_point = df1.iloc[ix - spike_loc:ix + spike_loc, :]
    df_point = df_point[target_col]
    df_point = df_point.sample(1, axis=1)
    point_replace = df1.iloc[[ix]]
    IQR = []
    spike_value = []
    spike_multiplier_range = (0.5, 3)

    for i in df_point.columns:
        percentile75 = df_point[i].quantile(0.25)
        percentile25 = df_point[i].quantile(0.75)
        iqr = percentile75 - percentile25
        IQR.append(iqr)
    for i in IQR:
        spike_value.append(random.uniform(i * spike_multiplier_range[0], i * spike_multiplier_range[1]))
    df_point.loc[point_replace.index] = spike_value
    df_point['label'] = 1
    df_point.loc[df_point.index, target_col] = df_point
    df.loc[df_point.index, target_col] = df_point
    df.loc[df_point.index, 'label'] = 1
    df.loc[df_point.index, 'synthetic_label'] = 1

    return df, df_point

# plot contextual and point anomaly
def plot_synthetic_data(feature, target_col, df_third_context, df_sec_point, df_swapped_1, df_swapped_2,
                        df_swapped_3, df_point_1, df_point_2, filename):
    df_third_context = df_third_context[target_col]

    n_features = df_third_context.shape[1]

    if n_features > 1:
        fig, ax = plt.subplots(n_features, 1, figsize=(10, n_features))
        for i, j in zip(range(n_features), df_third_context.columns):
            ax[i].plot(df_sec_point[j])
            ax[i].plot(df_swapped_1[j], color='red', alpha=0.6)
            ax[i].plot(df_swapped_2[j], color='red', alpha=0.6)
            ax[i].plot(df_swapped_3[j], color='red', alpha=0.6)
            ax[i].plot(df_point_1[j], color='orange')
            ax[i].plot(df_point_2[j], color='orange')
    else:
        fig = plt.figure(figsize=(15, 6))
        plt.plot(df_third_context.column[0])

    fig.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()

    if n_features > 1:
        fig, ax = plt.subplots(n_features, 1, figsize=(15, n_features))
        for i, j in zip(range(n_features), feature.columns):
            ax[i].plot(feature[j])
            ax[i].plot(feature[j], color='g', alpha=0.6)
    else:
        fig = plt.figure(figsize=(15, 6))
        plt.plot(df_third_context.column[0])

    plt.close('all')

    return df_third_context


def Chemei_building():
    global target_col, source_col

    df_chemei = pq.read_pandas('Chemie_single_testandtrain.parquet').to_pandas()
    df_chemei = df_chemei.set_index('Time')
    df_chemei = df_chemei.sort_index()
    df_chemei = df_chemei.drop(['Dayofweek', 'Hourofday', 'Normalize_timestamp', 'Day'], axis=1)

    source_col = ['6_Dfluss', '6_WTarif1_2', '6_Dfluss_2', '6_Vol_2', '6_Vorlauftmp_2', '6_Rücklauftmp_2',
                  '6_TmpDiff_2', '6_Wleistung_2', '1_BV+T1', '1_WV+T1']
    target_col = ['6_TmpDiff', '6_Rücklauftmp', '6_Vorlauftmp']

    df_chemei, df_feature = normalize(df_chemei)

    # ------------------ Generating Contextual and Point Anomaly --------------------------
    source1, target1, df1 = synthetic_anomaly(df_chemei)
    df_first_context, df_swapped_1 = contextual_anomaly(source1, target1, df1)
    df_first_point, df_point_1 = point_outlier(source1, target1, df_first_context)

    df_feature.loc[df_point_1.index, df_point_1.columns[0]] = 1
    df_feature.loc[df_point_1.index, 'label'] = 1

    source2, target2, df2 = synthetic_anomaly(df_first_point)
    df_sec_context, df_swapped_2 = contextual_anomaly(source2, target2, df2)
    df_sec_point, df_point_2 = point_outlier(source2, target2, df_sec_context)

    source3, target3, df3 = synthetic_anomaly(df_sec_point)
    df_third_context, df_swapped_3 = contextual_anomaly(source3, target3, df3)

    df_feature = df_feature[target_col]

    df_feature.loc[df_point_2.index, df_point_2.columns[0]] = 1
    df_feature.loc[df_point_2.index, 'label'] = 1

    df_feature.loc[target1.index, target1.columns[0]] = 1
    df_feature.loc[target1.index, 'label'] = 1

    df_feature.loc[target2.index, target2.columns[0]] = 1
    df_feature.loc[target2.index, 'label'] = 1

    df_feature.loc[target3.index, target3.columns[0]] = 1
    df_feature.loc[target3.index, 'label'] = 1

    plot_synthetic_data(df_feature, target_col, df_third_context, df_sec_point, df_swapped_1, df_swapped_2,
                        df_swapped_3,
                        df_point_1, df_point_2, 'Chemei_Building')

    return df_third_context


def EF42_building():
    global target_col, source_col

    df_EF42 = pq.read_pandas('EF42_single_testandtrain.parquet').to_pandas()
    df_EF42 = df_EF42.set_index('Time')
    df_EF42 = df_EF42.sort_index()
    df_EF42 = df_EF42.drop(['Dayofweek', 'Hourofday', 'Normalize_timestamp', 'Day'], axis=1)

    target_col = ['6_CN33 11 01 01_WTarif1', '6_CN33 11 01 01_Wleistung', '1_CN33 71 02 03_WV+T1']
    source_col = ['6_CN33 21 01 01_Dfluss', '6_CN33 11 01 01_Vorlauftmp', '6_CN33 11 01 01_Rücklauftmp',
                   '1_CN33 71 02 03_BV-T1']

    df_EF42, df_feature = normalize(df_EF42)

    # ------------------ Generating Contextual and Point Anomaly --------------------------
    source1, target1, df1 = synthetic_anomaly(df_EF42)
    df_first_context, df_swapped_1 = contextual_anomaly(source1, target1, df1)
    df_first_point, df_point_1 = point_outlier(source1, target1, df_first_context)

    df_feature.loc[df_point_1.index, df_point_1.columns[0]] = 1
    df_feature.loc[df_point_1.index, 'label'] = 1

    source2, target2, df2 = synthetic_anomaly(df_first_point)
    df_sec_context, df_swapped_2 = contextual_anomaly(source2, target2, df2)
    df_sec_point, df_point_2 = point_outlier(source2, target2, df_sec_context)

    source3, target3, df3 = synthetic_anomaly(df_sec_point)
    df_third_context, df_swapped_3 = contextual_anomaly(source3, target3, df3)

    df_feature = df_feature[target_col]

    df_feature.loc[df_point_2.index, df_point_2.columns[0]] = 1
    df_feature.loc[df_point_2.index, 'label'] = 1

    df_feature.loc[target1.index, target1.columns[0]] = 1
    df_feature.loc[target1.index, 'label'] = 1

    df_feature.loc[target2.index, target2.columns[0]] = 1
    df_feature.loc[target2.index, 'label'] = 1

    df_feature.loc[target3.index, target3.columns[0]] = 1
    df_feature.loc[target3.index, 'label'] = 1

    plot_synthetic_data(df_feature, target_col, df_third_context, df_sec_point, df_swapped_1, df_swapped_2,
                        df_swapped_3,
                        df_point_1, df_point_2, 'EF42_Building')

    return df_third_context


def main():
    df_chemie = Chemei_building()

    df_chemie.to_parquet('Synthetic_dataset_Chemei.parquet')

    df_EF42 = EF42_building()

    df_EF42.to_parquet('Synthetic_dataset_EF42.parquet')

if __name__ == '__main__':
    main()