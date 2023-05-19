import numpy as np
import pandas as pd

from typing import Tuple

from sklearn.model_selection import train_test_split


def split_dataframe(
    df: pd.DataFrame, train_ratio: float, val_ratio: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data segments of dataframe to the training, validation, and test segments.
    Author: Seongjae Lee

    Parameters
    ----------
    df: pd.DataFrame
        Input dataframe. The dataframe must contain the columns "data" and "label".
    train_ratio: float
        The ratio of the train data segment.
    val_ratio: float
        The ratio of the validation data segment.
        The test data segment ratio is automatically selected to 1 - train_ratio - val_ratio.
        train_ratio + val_ratio cannot be exceed 1.0.

    Returns
    ----------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        A tuple containing the train, validation, and test dataframes.
    """
    cum_train_ratio = train_ratio
    cum_val_ratio = cum_train_ratio + val_ratio

    train_df = {"data": [], "label": []}

    val_df = {"data": [], "label": []}

    test_df = {"data": [], "label": []}

    for _, row in df.iterrows():
        segment_length = row.data.size
        train_idx = (int)(segment_length * cum_train_ratio)
        val_idx = (int)(segment_length * cum_val_ratio)
        train_df["data"].append(row.data[:train_idx])
        train_df["label"].append(row.label)

        val_df["data"].append(row.data[train_idx:val_idx])
        val_df["label"].append(row.label)

        test_df["data"].append(row.data[val_idx:])
        test_df["label"].append(row.label)

    train_df = pd.DataFrame(train_df)
    val_df = pd.DataFrame(val_df)
    test_df = pd.DataFrame(test_df)

    return train_df, val_df, test_df


def build_from_dataframe(
    df: pd.DataFrame, sample_length: int, shift: int, one_hot: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate pairs of np.ndarrays from a dataframe.
    Author: Seongjae Lee

    Parameters
    ----------
    df: pd.DataFrame
        Input dataframe. The dataframe must contain the "data" column of np.ndarray type
        and the "label" column of int type.
    sample_length: int
        Length of samples for the input data.
    shift: int
        Interval between each sample when using overlapping to sample data.
    one_hot: bool, default False
        Whether to generate one-hot encoding for the data.

    Returns
    ----------
    Tuple[np.ndarray, np.ndarray]
        A tuple of two np.ndarray. Each array is the input and label of the dataset.
    """
    n_class = df["label"].max() - df["label"].min() + 1
    n_data = df.shape[0]
    data = []
    label = []
    for i in range(n_data):
        d = df.iloc[i]["data"]
        td, tl = sample_data(
            d, sample_length, shift, df.iloc[i]["label"], n_class, one_hot
        )
        data.append(td)
        label.append(tl)

    data_array = np.concatenate(tuple(data), axis=0)
    label_array = np.concatenate(tuple(label), axis=0)

    return data_array, label_array


def sample_data(
    data: np.ndarray,
    sample_length: int,
    shift: int,
    cls_id: int,
    num_class: int,
    one_hot: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate samples from the data segment.
    Author: Seongjae Lee

    Parameters
    ----------
    data: np.ndarray
        Input data segment.
    sample_length: int
        Length of samples for the input data.
    shift: int
        Interval between each sample when using overlapping to sample data.
    cls_id: int
        The class ID of the data.
    num_class: int
        The number of classes in the entire dataset
        (used only when creating one-hot encoding).
    one_hot: bool
        Whether to return the data in one-hot encoding.

    Returns
    ----------
    Tuple[np.ndarray, np.ndarray]
        Returns a tuple of (data, label).

    Raises
    ----------
    ValueError
        If the class ID is greater than the number of classes.
    """
    if cls_id >= num_class:
        raise ValueError("class id is out of bound")
    sampled_data = np.array(
        [
            data[i : i + sample_length]
            for i in range(0, len(data) - sample_length, shift)
        ]
    )
    if one_hot:
        label = np.zeros((sampled_data.shape[0], num_class))
        label[:, cls_id] = 1
    else:
        label = np.zeros((sampled_data.shape[0]))
        label = label + cls_id
    return sampled_data, label


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    val_size: float,
    train_size: float,
    random_state: int = None,
    shuffle: bool = True,
    stratify: np.ndarray = False,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """Split numpy array-style data pair (X, y) to train, validation, and test dataset.
    Author: Seongjae Lee

    Parameters
    ----------
    X: np.ndarray
        Data
    y: np.ndarray
        Lable
    test_size: float
        Ratio of the test dataset (0~1)
    val_size: float
        Ratio of the validation dataset (0~1)
    train_size: float
        Ratio of the train dataset (0~1)
    random_state: int
        Random state used for data split
    shuffle: bool
        Whether or not to shuffle the data before splitting.
    stratify: bool
        Option for the stratified split. If true, data is splited based on the label's distribution

    Returns
    ----------
    Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]
        Return ((X_train, y_train), (X_val, y_val), (X_test, y_test)) pairs.

    Raises
    ----------
        train_size + val_size + test size must be 1.0.

    """
    if train_size + val_size + test_size != 1.0:
        raise ValueError("data split ratio error")

    if stratify:
        stratify_y = y

    X_nt, X_test, y_nt, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=stratify_y,
    )

    if stratify:
        stratify_y = y_nt

    X_train, X_val, y_train, y_val = train_test_split(
        X_nt,
        y_nt,
        test_size=(val_size / (train_size + val_size)),
        random_state=random_state,
        shuffle=shuffle,
        stratify=stratify_y,
    )

    return ((X_train, y_train), (X_val, y_val), (X_test, y_test))
