import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        if train:
            X_ = []
            y_ = []
            for i in range(1, 6):
                file_path = os.path.join(base_folder, f"data_batch_{i}")
                with open(file_path, "rb") as file:
                    d = pickle.load(file, encoding="bytes")
                    X_.append(d[b"data"])
                    y_.append(d[b"labels"])
            X = np.concatenate(X_, axis=0)
            y = np.concatenate(y_, axis=0)
        else:
            file_path = os.path.join(base_folder, "test_batch")
            with open(file_path, "rb") as file:
                d = pickle.load(file, encoding="bytes")
            X, y = d[b"data"], d[b"labels"]
        X = X.astype(np.float32) / 255.
        self.X = X.reshape((-1, 3, 32, 32))
        self.y = np.asarray(y, dtype=np.uint8)
        self.transforms = transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        return self.apply_transforms(self.X[index]), self.y[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return self.y.shape[0]
        ### END YOUR SOLUTION
