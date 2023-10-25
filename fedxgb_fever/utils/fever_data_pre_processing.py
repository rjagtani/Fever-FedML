from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from typing import Dict, Tuple, Union
from flwr.common import NDArray
from config.config import partition_column, partition_col_index
import numpy as np

def fever_preprocess_split(data, test_size):
    label_encoder = LabelEncoder()
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = label_encoder.fit_transform(data[col])


    X = data.drop('quantity(kWh)', axis=1)
    y = data['quantity(kWh)']

    partition_col_index[0] = data.columns.get_loc(partition_column)  # This stores the index of the column based on which data will be partitioned

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    print("Size of the train set:", len(X_train))
    print("Size of the test set:", len(X_test))

    #Converting to Numpy Arrays
    X_train = X_train.values
    y_train = y_train.values
    X_test = X_test.values
    y_test = y_test.values


    X_train.flags.writeable = True
    y_train.flags.writeable = True
    X_test.flags.writeable = True
    y_test.flags.writeable = True

    print("Feature dimension of the dataset:", X_train.shape[1])
    print("Size of the trainset:", X_train.shape[0])
    print("Size of the testset:", X_test.shape[0])
    assert X_train.shape[1] == X_test.shape[1]

    return(X_train, y_train, X_test, y_test)


def get_dataloader(
    dataset: Dataset, partition: str, batch_size: Union[int, str]
) -> DataLoader:
    if batch_size == "whole":
        batch_size = len(dataset)
    return DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, shuffle=(partition == "train")
    )


# https://github.com/adap/flower
def do_fl_partitioning(
    trainset: Dataset,
    testset: Dataset,
    pool_size: int,
    batch_size: Union[int, str],
    val_ratio: float = 0.0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    partition_unique = sorted(np.unique(trainset.data[:, partition_col_index[0]]))
    if len(partition_unique) < pool_size:
        raise ValueError("The number of unique feature values are less than the required client pool size.")
    datasets = []
    for values in partition_unique[:pool_size]:
        partition_indices = np.where(trainset.data[:, partition_col_index[0]] == values)[0]
        dataset = Subset(trainset, partition_indices)
        datasets.append(dataset)

    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for ds in datasets:
        len_val = int(len(ds) * val_ratio)
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(0))
        trainloaders.append(get_dataloader(ds_train, "train", batch_size))
        if len_val != 0:
            valloaders.append(get_dataloader(ds_val, "val", batch_size))
        else:
            valloaders = None
    testloader = get_dataloader(testset, "test", batch_size)
    return trainloaders, valloaders, testloader



