import torch
import numpy as np
import torchvision
from torchvision.datasets import MNIST, CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import tensorflow as tf


# Load MNIST

def preprocess_mnist(images):
    images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
    return np.where(images > .5, 1.0, 0.0).astype('float32')

def load_static_mnist(bs):
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    x_train = preprocess_mnist(train_images)
    x_test = preprocess_mnist(test_images)
    x_train = np.reshape(x_train, (-1, 28*28))
    x_test = np.reshape(x_test, (-1, 28*28))
    y_train = tf.one_hot(train_labels, 10).numpy()
    y_test = tf.one_hot(test_labels, 10).numpy()
    train = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = torch.utils.data.DataLoader(train, batch_size=bs, shuffle=True)
    validation = torch.utils.data.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    val_loader = torch.utils.data.DataLoader(validation, batch_size=bs, shuffle=False)
    test = torch.utils.data.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = torch.utils.data.DataLoader(test, batch_size=bs, shuffle=True)
    return train_loader, val_loader, test_loader


def remove_labels(x, y, rm_labels):

    keep_indices = []

    for i in range(len(y)):
        if y[i] not in rm_labels:
            keep_indices.append(i)

    x = x[keep_indices]
    y = y[keep_indices]

    return x, y

def load_cifar10(batch_size, rm_labels=0):
    # set args
    input_size = [3, 32, 32]
    input_type = 'continuous'
    dynamic_binarization = False

    # start processing
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # load main train dataset
    training_dataset = CIFAR10('datasets/Cifar10/', train=True, download=True, transform=transform)
    train_data = np.clip((training_dataset.data + 0.5) / 256., 0., 1.)
    train_data = np.swapaxes( np.swapaxes(train_data,1,2), 1, 3)
    train_data = np.reshape(train_data, (-1, np.prod(input_size)) )


    x_val = train_data[int(50000*0.8/10*(10-rm_labels)):50000]
    x_train = train_data[0:int(50000*0.8/10*(10-rm_labels))]

    y_train = np.array(training_dataset.targets)[0:int(50000*0.8/10*(10-rm_labels))]
    y_val = np.array(training_dataset.targets)[int(50000*0.8/10*(10-rm_labels)):50000]

    if rm_labels == 4:
        x_train, y_train = remove_labels(x_train, y_train, [0,2,3,4])
        x_val, y_val = remove_labels(x_val, y_val, [0,2,3,4])
    elif rm_labels == 2:
        x_train, y_train = remove_labels(x_train, y_train, [3,4])
        x_val, y_val = remove_labels(x_val, y_val, [3,4])

    # train loader
    train = torch.utils.data.TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

    # validation loader
    validation = torch.utils.data.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    val_loader = torch.utils.data.DataLoader(validation, batch_size=batch_size, shuffle=False)

    # test loader
    test_dataset = CIFAR10('datasets/Cifar10/', train=False, transform=transform )
    test_data = np.clip((test_dataset.data + 0.5) / 256., 0., 1.)
    test_data = np.swapaxes( np.swapaxes(test_data,1,2), 1, 3)
    x_test = np.reshape(test_data, (-1, np.prod(input_size)) )
    y_test = np.array(test_dataset.targets)

    if rm_labels == 4:
        x_test, y_test = remove_labels(x_test, y_test, [0,2,3,4])
    elif rm_labels == 2:
        x_test, y_test = remove_labels(x_test, y_test, [3,4])

    test = torch.utils.data.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader