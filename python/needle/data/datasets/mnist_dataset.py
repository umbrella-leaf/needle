from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip
import struct

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        with gzip.open(image_filename, "rb") as image_path:
          magic, num, self.row, self.col = struct.unpack(">IIII", image_path.read(16))
          images = np.fromstring(image_path.read(), dtype=np.uint8).reshape(num, self.row*self.col).astype(np.float32)
          images /= 255.0
        with gzip.open(label_filename, "rb") as label_path:
          magic, num = struct.unpack(">II", label_path.read(8))
          labels = np.fromstring(label_path.read(), dtype=np.uint8)
        self.images = images
        self.labels = labels
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        image = self.images[index]
        label = self.labels[index]
        if self.transforms:
          image = self.apply_transforms(image.reshape(self.row, self.col, -1))
          image = image.reshape(-1, self.row*self.col)
        return (image, label)
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.images.shape[0]
        ### END YOUR SOLUTION