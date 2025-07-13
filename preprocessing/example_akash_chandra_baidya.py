import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.impute import SimpleImputer

def _load_data(file_path, columns):
    # tbd: load the parquet file from file_path and return as pandas DataFrame
    data=pd.read_parquet(file_path, engine="pyarrow")

    return data


# a=_load_data("D:\\5th Sem\\dataframe.parquet",1)
# print(a)


def _train_test_split(data, split_rate=0.4):
    # split the data into training and testing sets based on the split_rate parameter
    train_set, test_set=train_test_split(data,test_size=split_rate)
    return train_set, test_set



# train_set, test_set = _train_test_split(a, split_rate=0.4)
# print(test_set)

def _fill_missing_values(dataset):
    # fill missing values (pay attention to label column)
    # print(dataset)
    filled_dataset=dataset.fillna(0.0)
    # filled_dataset=dataset.fillna(method="ffill",axis=0)
    # filled_dataset=dataset.copy()
    # print(dataset)
    # dataset.replace('?',np.NaN,inplace=True)
    # imp=SimpleImputer(missing_values=np.NaN)
    # idf=pd.DataFrame(imp.fit_transform(dataset.loc[:,dataset.columns!="OBIS Bezeichnung"]))
    # print(idf)
    
    # dataset.loc[:,dataset.columns!="OBIS Bezeichnung"]=idf

    
    # idf.columns=df.columns
    # idf.index=df.index

    # idf['bare_nuclei'].isna().sum()

    
    # simple 
    return filled_dataset

# x=_fill_missing_values(a)
# print(x)
# # # x.info()
# # x.isnull().sum()
# # x.any()

def _add_artificial_label(dataset):
    # manually label the dataset and append the label as the last column
    dataset.insert(21,"Label",1)
    # dataset["Label"]=1
    labeled_dataset=dataset
    # print(labeled_dataset)
    return labeled_dataset


# s=_add_artificial_label(x)
# s
# s.info()
# s.isna().sum()
# s.any()
# s
def _preprocessing(subset):

    # # apply multiple preprocessing approaches, e.g. remove trend, normalization

    #normalization
    preprocessed_dataset= (subset.loc[:,subset.columns!="OBIS Bezeichnung"]-subset.loc[:,subset.columns!="OBIS Bezeichnung"].mean())/subset.loc[:,subset.columns!="OBIS Bezeichnung"].std()
    preprocessed_dataset.insert(0,"OBIS Bezeichnung",subset["OBIS Bezeichnung"])
    preprocessed_dataset=preprocessed_dataset.iloc[:,0:]


    # scaler=preprocessing.MinMaxScaler()

    # d=scaler.fit_transform(subset.columns!=["OBIS Bezeichnung"])

    # # d=preprocessing.normalize(subset.loc[:,subset.columns!="OBIS Bezeichnung"],axis=0)
    # # housing = pd.read_csv("/content/sample_data/california_housing_train.csv")
    # d = preprocessing.normalize(subset.loc[:,subset.columns!="OBIS Bezeichnung"], axis=0)
    # scaled_df = pd.DataFrame(d, columns=names)
    # scaled_df.head()

    
    # # print(d)
    
    # # names=subset.columns
    # # scaled_subset=pd.DataFrame(d)
    # # preprocessed_dataset=scaled_subset
    # print(preprocessed_dataset)

    return preprocessed_dataset

# o=_preprocessing(x)
# # s.describe()
# # o.describe()


def get_data(file_path, fill_missing_value=True):
    data = _load_data(file_path,1) 
    labeled_dataset = _add_artificial_label(data)
    if fill_missing_value:
        labeled_dataset = _fill_missing_values(labeled_dataset)
    
    train_set, test_set = _train_test_split(labeled_dataset, split_rate=0.4)
    # print(train_set)
    preprocessed_train_set, preprocessed_test_set = _preprocessing(train_set), _preprocessing(test_set)
    print(preprocessed_train_set)
    # print(preprocessed_train_set.describe())
    return preprocessed_train_set.iloc[:, :-1], \
           preprocessed_train_set.iloc[:, -1], \
           preprocessed_test_set.iloc[:, :-1], \
           preprocessed_test_set.iloc[:, -1]

get_data("D:\\5th Sem\\dataframe.parquet")
