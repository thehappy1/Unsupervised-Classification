"""
This code is based on the Torchvision repository, which was licensed under the BSD 3-Clause.
"""
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd


class Fashion(Dataset):
    """User defined class to build a datset using Pytorch class Dataset."""

    def __init__(self, train=True, transform=None):
        """Method to initilaize variables."""

        self.train = train
        self.transform = transform

        if self.train:
            train_csv = pd.read_csv("FashionMNIST/FashionMNIST/csv/fashion-mnist_train.csv")
            self.fashion_MNIST = list(train_csv.values)
        else:
            test_csv = pd.read_csv("FashionMNIST/FashionMNIST/csv/fashion-mnist_test.csv")
            self.fashion_MNIST = list(test_csv.values)

        label = []
        image = []

        for i in self.fashion_MNIST:
            # first column is of labels.
            label.append(i[0])
            image.append(i[1:])
        self.labels = np.asarray(label)

        # Dimension of Images = 28 * 28 * 1. where height = width = 28 and color_channels = 1.
        self.images = np.asarray(image).reshape(-1, 28, 28).astype('uint8')

        #repeat values of the 1-dimensional image 3 times to get rgb format
        self.images = np.repeat(self.images[..., np.newaxis], 3, -1)

    def __getitem__(self, index):
        label = self.labels[index]
        image = self.images[index]

        #Return a PIL image
        image = Image.fromarray(image)
        img_size = image.size

        if self.transform is not None:
            image = self.transform(image)

        out = {'image': image, 'target': label, 'meta': {'im_size': img_size, 'index': index}}

        return out

    def __len__(self):
        return len(self.images)