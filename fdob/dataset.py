import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from typing import Tuple


class NumpyDataset(Dataset):
    """
    PyTorch Dataset for numpy array-style dataset.
    Author: Seongjae Lee

    Attributes
    ----------
    data: np.ndarray
        Numpy array-style data. shape: [N, ...]
    label: np.ndarray
        Numpy array-style label. shape: [N,]
    transfrom: torchvision.transforms.transforms.Compose
        Data transform class. default: None
    target_transform: torchvision.transforms.transforms.Compose
        Label transform class. default: None

    Methods
    ----------
    __len__:
        Return the length of the data.
    __getitem__(idx):
        Return the "idx-th" index of the data.
    """

    def __init__(
        self,
        data: np.ndarray,
        label: np.ndarray,
        transform: transforms.transforms.Compose = None,
        target_transform: transforms.transforms.Compose = None,
    ) -> None:
        self.data = data
        self.label = label
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        x = self.data[idx, :].astype("float32")
        t = np.array(self.label[idx]).astype("int64")

        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            t = self.target_transform(t)

        return x, t


def get_dataloader(
    data: np.ndarray,
    label: np.ndarray,
    transform_data: transforms.transforms.Compose,
    transform_label: transforms.transforms.Compose,
    shuffle: bool,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    """Generate PyTorch DataLoader.
    Author: Seongjae Lee

    Parameters
    ----------
    data: np.ndarray
        Numpy array-style data. shape: [N, ...]
    label: np.ndarray
        Numpy array-style label. shape: [N,]
    transfrom_data: torchvision.transforms.transforms.Compose
        Data transform class.
    target_label: torchvision.transforms.transforms.Compose
        Label transform class.
    shuffle: bool
        Whether or not to shuffle the data when iterating.
    batch size: int
        Batch size for the model training.
    num_workers: int
        The number of processors fetching the data.

    Returns
    ----------
    DataLoader
        Return PyTorch DataLoader for training.
    """
    dataset = NumpyDataset(data, label, transform_data, transform_label)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
    )


class DatasetHandler:
    """
    Class that handles multi-domain dataset.
    Author: Seongjae Lee

    Attributes
    ----------
    dataloaders: Dict
        Python dictionary that contains multi-domain datasets.
        The keys of the dictionary represents data domain.
    """

    def __init__(self):
        self.dataloaders = {}

    def assign(
        self,
        data_train: np.ndarray,
        label_train: np.ndarray,
        data_val: np.ndarray,
        label_val: np.ndarray,
        data_test: np.ndarray,
        label_test: np.ndarray,
        sample_length: int,
        tag: str,
        transform_data: transforms.transforms.Compose,
        transform_label: transforms.transforms.Compose,
        batch_size: int,
        num_workers: int,
    ) -> None:
        """Assign dataset to domain ``tag''

        Parameters
        ----------
        data_train: np.ndarray
            Training data [N, sample_length]
        label_train: np.ndarray
            Training label [N, ]
        data_val: np.ndarray
            Validation data [N, sample_length]
        label_val: np.ndarray
            Validation label [N, ]
        data_test: np.ndarray
            Test data [N, sample_length]
        label_test: np.ndarray
            Test label [N, ]
        sample_length: int
            A sample length of datasets. The sample length of all
            datasets must be same.
        tag: str
            Domain tag.
        transfrom_data: torchvision.transforms.transforms.Compose
            Data transform class.
        target_label: torchvision.transforms.transforms.Compose
            Label transform class.
        batch size: int
            Batch size for the model training.
        num_workers: int
            The number of processors fetching the data.

        Returns
        ----------
            No return Values. Just add the dataset to self.dataloaders
        """
        if data_train.shape[1] < sample_length:
            raise ValueError("data length is not compatible for sample length")
        if data_val.shape[1] < sample_length:
            raise ValueError("data length is not compatible for sample length")
        if data_test.shape[1] < sample_length:
            raise ValueError("data length is not compatible for sample length")

        data_train = data_train[:, :sample_length]
        data_val = data_val[:, :sample_length]
        data_test = data_test[:, :sample_length]

        train_loader = get_dataloader(
            data_train,
            label_train,
            transform_data,
            transform_label,
            True,
            batch_size,
            num_workers,
        )
        val_loader = get_dataloader(
            data_val,
            label_val,
            transform_data,
            transform_label,
            False,
            batch_size,
            num_workers,
        )
        test_loader = get_dataloader(
            data_test,
            label_test,
            transform_data,
            transform_label,
            False,
            batch_size,
            num_workers,
        )

        loaders = {"train": train_loader, "val": val_loader, "test": test_loader}

        self.dataloaders[tag] = loaders
