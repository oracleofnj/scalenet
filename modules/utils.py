"""Utilities for scalenet."""

import pickle
import os
import numpy as np

BASE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
    "image_files"
)


def load_cifar10(num_batches=5,
                 get_test_data=True,
                 channels_last=True):
    """Load the cifar data.

    Args:
        num_batches: int, the number of batches of data to return
        get_test_data: bool, whether to return test data
    Returns:
        (images, labels) it get_test_data False
        (images, labels, test_images, test_labels) otherwise
        images are numpy arrays of shape:
                    (num_images, num_channels, width, height)
        labels are 1D numpy arrays
    """
    assert num_batches <= 5

    # load batches in order:
    dirpath = os.path.join(BASE_DIR, 'cifar-10-batches-py')
    images = None
    for i in range(1, num_batches + 1):
        print('getting batch {0}'.format(i))
        filename = 'data_batch_{0}'.format(i)
        fpath = os.path.join(dirpath, filename)
        with open(fpath, 'rb') as f:
            content = pickle.load(f, encoding='bytes')
        if images is None:
            images = content[b'data']
            labels = content[b'labels']
        else:
            images = np.vstack([images, content[b'data']])
            labels.extend(content[b'labels'])
    # convert to labels:
    labels = np.asarray(labels)
    # convert to RGB format:
    images = images.reshape(-1, 3, 32, 32)

    # normalize data by dividing by 255:
    images = images / 255.
    if channels_last:
        images = np.moveaxis(images, 1, -1)

    if not get_test_data:
        return images, labels

    filename = 'test_batch'
    fpath = os.path.join(dirpath, filename)
    with open(fpath, 'rb') as f:
        content = pickle.load(f, encoding='bytes')
    test_images = content[b'data'].reshape(-1, 3, 32, 32)
    test_labels = np.asarray(content[b'labels'])

    # normalize:
    test_images = test_images / 255.
    # make channels last:
    if channels_last:
        test_images = np.moveaxis(test_images, 1, -1)

    return images, labels, test_images, test_labels


def load_cifar100(get_test_data=True,
                  channels_last=True):
    """Load the cifar 100 data (not in batches).

    Args:
        get_test_data: bool, whether to return test data
    Returns:
        (images, labels) it get_test_data False
        (images, labels, test_images, test_labels) otherwise
        images are numpy arrays of shape:
                    (num_images, num_channels, width, height)
        labels are 1D numpy arrays
    """
    # load batches in order:
    dirpath = os.path.join(BASE_DIR, 'cifar-100-python')
    images = None
    filename = 'train'
    fpath = os.path.join(dirpath, filename)
    with open(fpath, 'rb') as f:
        content = pickle.load(f, encoding='bytes')
    if images is None:
        images = content[b'data']
        labels = content[b'fine_labels']
    # convert to labels:
    labels = np.asarray(labels)
    # convert to RGB format:
    images = images.reshape(-1, 3, 32, 32)

    # normalize data by dividing by 255:
    images = images / 255.
    if channels_last:
        images = np.moveaxis(images, 1, -1)

    if not get_test_data:
        return images, labels

    filename = 'test'
    fpath = os.path.join(dirpath, filename)
    with open(fpath, 'rb') as f:
        content = pickle.load(f, encoding='bytes')
    test_images = content[b'data'].reshape(-1, 3, 32, 32)
    test_labels = np.asarray(content[b'fine_labels'])

    # normalize:
    test_images = test_images / 255.
    # make channels last:
    if channels_last:
        test_images = np.moveaxis(test_images, 1, -1)

    return images, labels, test_images, test_labels
