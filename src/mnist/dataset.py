import os

from torch.utils.data import DataLoader
import torch
from torchvision import datasets, transforms


def get(batch_size, data_root='/tmp/public_dataset/pytorch', train=True, val=True, fix_shuffle=False, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'mnist-data'))
    kwargs.pop('input_size', None)
    num_workers = kwargs.setdefault('num_workers', 1)
    print("Building MNIST data loader with {} workers".format(num_workers))
    ds = []
    if train:
        shuffle = False if fix_shuffle else True
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(root=data_root, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               #transforms.Normalize((0.5,), (0.5,))
                           ])),
            batch_size=batch_size, shuffle=shuffle, **kwargs)
        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(root=data_root, train=False, download=True,
                           transform=transforms.Compose([
                                transforms.ToTensor(),
                                #transforms.Normalize((0.5,), (0.5,))
                            ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds
