from __future__ import print_function

import torch
import torch.utils.data as data_utils
import torchvision

import tensorflow as tf
import numpy as np

from scipy.io import loadmat
import os

import pickle
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# ======================================================================================================================


# Helper functions
def remove_labels(x, y, rm_labels):

    keep_indices = []

    for i in range(len(y)):
        if y[i] not in rm_labels:
            keep_indices.append[i]

    x = x[keep_indices]
    y = y[keep_indices]

    return x, y

def create_flips(epsilon, data, digit_dist=False):
    if digit_dist == False:
      flips_matrix = [[-np.random.binomial(1,epsilon) for i in range(784)] for n in range(len(data))]
    else:
      flips_matrix = []
      for data_point in data:
        flips_matrix.append(-np.random.binomial(1,epsilon*(np.absolute(data_point-digit_dist[np.random.randint(0, 10)]))))
    return np.array(flips_matrix)


def preprocess_mnist(images):
    images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
    return np.where(images > .5, 1.0, 0.0).astype('float32')
    

def load_static_mnist(args, **kwargs):
    # set args
    args.input_size = [1, 28, 28]
    args.input_type = 'binary' # or binary
    args.dynamic_binarization = False
    
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    x_train = preprocess_mnist(train_images)
    x_test = preprocess_mnist(test_images)
    x_train = np.reshape(x_train, (-1, 28*28))
    x_test = np.reshape(x_test, (-1, 28*28))
    y_train = tf.one_hot(train_labels, 10).numpy()
    y_test = tf.one_hot(test_labels, 10).numpy()


    #add noise to train and test data

    #find distribution over each digit/class
    #digit_dists_train = digit_distributions(x_train, train_labels) 
    #digit_dists_test = digit_distributions(x_test, test_labels)
    
    if args.add_noise:
        epsilon = args.epsilon
        print("Epsilon = ", epsilon)
        x_train = np.float32(np.absolute(x_train + create_flips(epsilon, x_train, digit_dist=not args.random_noise)))
        x_test = np.float32(np.absolute(x_test + create_flips(epsilon, x_test, digit_dist=not args.random_noise)))
        

    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True, **kwargs)

    validation = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    val_loader = data_utils.DataLoader(validation, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # setting pseudo-inputs inits
    if args.use_training_data_init == 1:
        args.pseudoinputs_std = 0.01
        init = x_train[0:args.number_components].T
        args.pseudoinputs_mean = torch.from_numpy( init + args.pseudoinputs_std * np.random.randn(np.prod(args.input_size), args.number_components) ).float()
    else:
        args.pseudoinputs_mean = 0.05
        args.pseudoinputs_std = 0.01

    return train_loader, val_loader, test_loader, args

# ======================================================================================================================
def load_dynamic_mnist(args, **kwargs):
    # set args
    args.input_size = [1, 28, 28]
    args.input_type = 'binary'
    args.dynamic_binarization = True

    # start processing
    from torchvision import datasets, transforms
    train_loader = torch.utils.data.DataLoader( datasets.MNIST('../data', train=True, download=True,
                                                               transform=transforms.Compose([
                                                                   transforms.ToTensor()
                                                               ])),
                                                batch_size=args.batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader( datasets.MNIST('../data', train=False,
                                                              transform=transforms.Compose([transforms.ToTensor()
                                                                        ])),
                                               batch_size=args.batch_size, shuffle=True)

    # preparing data
    x_train = train_loader.dataset.train_data.float().numpy() / 255.
    x_train = np.reshape( x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2] ) )

    y_train = np.array( train_loader.dataset.train_labels.float().numpy(), dtype=int)

    x_test = test_loader.dataset.test_data.float().numpy() / 255.
    x_test = np.reshape( x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2] ) )

    y_test = np.array( test_loader.dataset.test_labels.float().numpy(), dtype=int)

    # validation set
    x_val = x_train[50000:60000]
    y_val = np.array(y_train[50000:60000], dtype=int)
    x_train = x_train[0:50000]
    y_train = np.array(y_train[0:50000], dtype=int)

    # binarize
    if args.dynamic_binarization:
        args.input_type = 'binary'
        np.random.seed(777)
        x_val = np.random.binomial(1, x_val)
        x_test = np.random.binomial(1, x_test)
    else:
        args.input_type = 'gray'

    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True, **kwargs)

    validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    val_loader = data_utils.DataLoader(validation, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    # setting pseudo-inputs inits
    if args.use_training_data_init == 1:
        args.pseudoinputs_std = 0.01
        init = x_train[0:args.number_components].T
        args.pseudoinputs_mean = torch.from_numpy( init + args.pseudoinputs_std * np.random.randn(np.prod(args.input_size), args.number_components) ).float()
    else:
        args.pseudoinputs_mean = 0.05
        args.pseudoinputs_std = 0.01

    return train_loader, val_loader, test_loader, args

# ======================================================================================================================
def load_omniglot(args, n_validation=1345, **kwargs):
    # set args
    args.input_size = [1, 28, 28]
    args.input_type = 'binary'
    args.dynamic_binarization = True

    # start processing
    def reshape_data(data):
        return data.reshape((-1, 28, 28)).reshape((-1, 28*28), order='F')
    omni_raw = loadmat(os.path.join('datasets', 'OMNIGLOT', 'chardata.mat'))

    # train and test data
    train_data = reshape_data(omni_raw['data'].T.astype('float32'))
    x_test = reshape_data(omni_raw['testdata'].T.astype('float32'))

    # shuffle train data
    np.random.shuffle(train_data)

    # set train and validation data
    x_train = train_data[:-n_validation]
    x_val = train_data[-n_validation:]

    # binarize
    if args.dynamic_binarization:
        args.input_type = 'binary'
        np.random.seed(777)
        x_val = np.random.binomial(1, x_val)
        x_test = np.random.binomial(1, x_test)
    else:
        args.input_type = 'gray'

    # idle y's
    y_train = np.zeros( (x_train.shape[0], 1) )
    y_val = np.zeros( (x_val.shape[0], 1) )
    y_test = np.zeros( (x_test.shape[0], 1) )

    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True, **kwargs)

    validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    val_loader = data_utils.DataLoader(validation, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # setting pseudo-inputs inits
    if args.use_training_data_init == 1:
        args.pseudoinputs_std = 0.01
        init = x_train[0:args.number_components].T
        args.pseudoinputs_mean = torch.from_numpy( init + args.pseudoinputs_std * np.random.randn(np.prod(args.input_size), args.number_components) ).float()
    else:
        args.pseudoinputs_mean = 0.05
        args.pseudoinputs_std = 0.01

    return train_loader, val_loader, test_loader, args

# ======================================================================================================================
def load_caltech101silhouettes(args, **kwargs):
    # set args
    args.input_size = [1, 28, 28]
    args.input_type = 'binary'
    args.dynamic_binarization = False

    # start processing
    def reshape_data(data):
        return data.reshape((-1, 28, 28)).reshape((-1, 28*28), order='fortran')
    caltech_raw = loadmat(os.path.join('datasets', 'Caltech101Silhouettes', 'caltech101_silhouettes_28_split1.mat'))

    # train, validation and test data
    x_train = 1. - reshape_data(caltech_raw['train_data'].astype('float32'))
    np.random.shuffle(x_train)
    x_val = 1. - reshape_data(caltech_raw['val_data'].astype('float32'))
    np.random.shuffle(x_val)
    x_test = 1. - reshape_data(caltech_raw['test_data'].astype('float32'))

    y_train = caltech_raw['train_labels']
    y_val = caltech_raw['val_labels']
    y_test = caltech_raw['test_labels']

    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True, **kwargs)

    validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    val_loader = data_utils.DataLoader(validation, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # setting pseudo-inputs inits
    if args.use_training_data_init == 1:
        args.pseudoinputs_std = 0.01
        init = x_train[0:args.number_components].T
        args.pseudoinputs_mean = torch.from_numpy( init + args.pseudoinputs_std * np.random.randn(np.prod(args.input_size), args.number_components) ).float()
    else:
        args.pseudoinputs_mean = 0.5
        args.pseudoinputs_std = 0.02

    return train_loader, val_loader, test_loader, args

# ======================================================================================================================
def load_histopathologyGray(args, **kwargs):
    # set args
    args.input_size = [1, 28, 28]
    args.input_type = 'gray'
    args.dynamic_binarization = False

    # start processing
    with open('datasets/HistopathologyGray/histopathology.pkl', 'rb') as f:
        data = pickle.load(f)

    x_train = np.asarray(data['training']).reshape(-1, 28 * 28)
    x_val = np.asarray(data['validation']).reshape(-1, 28 * 28)
    x_test = np.asarray(data['test']).reshape(-1, 28 * 28)

    x_train = np.clip(x_train, 1./512., 1. - 1./512.)
    x_val = np.clip(x_val, 1./512., 1. - 1./512.)
    x_test = np.clip(x_test, 1./512., 1. - 1./512.)

    # idle y's
    y_train = np.zeros( (x_train.shape[0], 1) )
    y_val = np.zeros( (x_val.shape[0], 1) )
    y_test = np.zeros( (x_test.shape[0], 1) )

    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True, **kwargs)

    validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    val_loader = data_utils.DataLoader(validation, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # setting pseudo-inputs inits
    if args.use_training_data_init == 1:
        args.pseudoinputs_std = 0.01
        init = x_train[0:args.number_components].T
        args.pseudoinputs_mean = torch.from_numpy( init + args.pseudoinputs_std * np.random.randn(np.prod(args.input_size), args.number_components) ).float()
    else:
        args.pseudoinputs_mean = 0.4
        args.pseudoinputs_std = 0.05

    return train_loader, val_loader, test_loader, args

# ======================================================================================================================
def load_freyfaces(args, TRAIN = 1565, VAL = 200, TEST = 200, **kwargs):
    # set args
    args.input_size = [1, 28, 20]
    args.input_type = 'gray'
    args.dynamic_binarization = False

    # start processing
    with open('datasets/Freyfaces/freyfaces.pkl', 'rb') as f:
        data = pickle.load(f)

    data = (data[0] + 0.5) / 256.

    # shuffle data:
    np.random.shuffle(data)

    # train images
    x_train = data[0:TRAIN].reshape(-1, 28*20)
    # validation images
    x_val = data[TRAIN:(TRAIN + VAL)].reshape(-1, 28*20)
    # test images
    x_test = data[(TRAIN + VAL):(TRAIN + VAL + TEST)].reshape(-1, 28*20)

    # idle y's
    y_train = np.zeros( (x_train.shape[0], 1) )
    y_val = np.zeros( (x_val.shape[0], 1) )
    y_test = np.zeros( (x_test.shape[0], 1) )

    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True, **kwargs)

    validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    val_loader = data_utils.DataLoader(validation, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # setting pseudo-inputs inits
    if args.use_training_data_init == 1:
        args.pseudoinputs_std = 0.01
        init = x_train[0:args.number_components].T
        args.pseudoinputs_mean = torch.from_numpy( init + args.pseudoinputs_std * np.random.randn(np.prod(args.input_size), args.number_components) ).float()
    else:
        args.pseudoinputs_mean = 0.5
        args.pseudoinputs_std = 0.02

    return train_loader, val_loader, test_loader, args

# ======================================================================================================================
def load_cifar10(args, **kwargs):
    # set args
    args.input_size = [3, 32, 32]
    args.input_type = 'continuous'
    args.dynamic_binarization = False

    # start processing
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # load main train dataset
    training_dataset = datasets.CIFAR10('datasets/Cifar10/', train=True, download=True, transform=transform)
    train_data = np.clip((training_dataset.data + 0.5) / 256., 0., 1.)
    train_data = np.swapaxes( np.swapaxes(train_data,1,2), 1, 3)
    train_data = np.reshape(train_data, (-1, np.prod(args.input_size)) )
    np.random.shuffle(train_data)

    x_val = train_data[40000:50000]
    x_train = train_data[0:40000]

    # fake labels just to fit the framework
    y_train = np.zeros( (x_train.shape[0], 1) )
    y_val = np.zeros( (x_val.shape[0], 1) )

    if args.rm_labels == 4:
        x_train, y_train = remove_labels(x_train, y_train, [0,2,3,4])
        x_val, y_val = remove_labels(x_val, y_val, [0,2,3,4])
    elif args.rm_labels == 2:
        x_train, y_train = remove_labels(x_train, y_train, [3,4])
        x_val, y_val = remove_labels(x_val, y_val, [3,4])

    # train loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True, **kwargs)

    # validation loader
    validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    val_loader = data_utils.DataLoader(validation, batch_size=args.test_batch_size, shuffle=False, **kwargs)


    # test loader
    test_dataset = datasets.CIFAR10('datasets/Cifar10/', train=False, transform=transform )
    test_data = np.clip((test_dataset.data + 0.5) / 256., 0., 1.)
    test_data = np.swapaxes( np.swapaxes(test_data,1,2), 1, 3)
    x_test = np.reshape(test_data, (-1, np.prod(args.input_size)) )

    y_test = np.zeros((x_test.shape[0], 1))

    if args.rm_labels == 4:
        x_test, y_test = remove_labels(x_test, y_test, [0,2,3,4])
    elif args.rm_labels == 2:
        x_test, y_test = remove_labels(x_test, y_test, [3,4])

    test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # setting pseudo-inputs inits
    if args.use_training_data_init == 1:
        args.pseudoinputs_std = 0.01
        init = x_train[0:args.number_components].T
        args.pseudoinputs_mean = torch.from_numpy( init + args.pseudoinputs_std * np.random.randn(np.prod(args.input_size), args.number_components) ).float()
    else:
        args.pseudoinputs_mean = 0.4
        args.pseudoinputs_std = 0.05

    return train_loader, val_loader, test_loader, args


def load_cifar100(args, **kwargs):
    # set args
    args.input_size = [3, 32, 32]
    args.input_type = 'continuous'
    args.dynamic_binarization = False

    # start processing
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # load main train dataset
    training_dataset = datasets.CIFAR100('datasets/Cifar100/', train=True, download=True, transform=transform)
    train_data = np.clip((training_dataset.data + 0.5) / 256., 0., 1.)
    train_data = np.swapaxes( np.swapaxes(train_data,1,2), 1, 3)
    train_data = np.reshape(train_data, (-1, np.prod(args.input_size)) )
    np.random.shuffle(train_data)

    x_val = train_data[40000:50000]
    x_train = train_data[0:40000]

    # fake labels just to fit the framework
    y_train = np.zeros( (x_train.shape[0], 1) )
    y_val = np.zeros( (x_val.shape[0], 1) )

    # train loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True, **kwargs)

    # validation loader
    validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    val_loader = data_utils.DataLoader(validation, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    # test loader
    test_dataset = datasets.CIFAR100('datasets/Cifar100/', train=False, transform=transform )
    test_data = np.clip((test_dataset.data + 0.5) / 256., 0., 1.)
    test_data = np.swapaxes( np.swapaxes(test_data,1,2), 1, 3)
    x_test = np.reshape(test_data, (-1, np.prod(args.input_size)) )

    y_test = np.zeros((x_test.shape[0], 1))

    test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # setting pseudo-inputs inits
    if args.use_training_data_init == 1:
        args.pseudoinputs_std = 0.01
        init = x_train[0:args.number_components].T
        args.pseudoinputs_mean = torch.from_numpy( init + args.pseudoinputs_std * np.random.randn(np.prod(args.input_size), args.number_components) ).float()
    else:
        args.pseudoinputs_mean = 0.4
        args.pseudoinputs_std = 0.05

    return train_loader, val_loader, test_loader, args

# ======================================================================================================================
def load_dataset(args, **kwargs):
    if args.dataset_name == 'static_mnist':
        train_loader, val_loader, test_loader, args = load_static_mnist(args, **kwargs)
    elif args.dataset_name == 'dynamic_mnist':
        train_loader, val_loader, test_loader, args = load_dynamic_mnist(args, **kwargs)
    elif args.dataset_name == 'omniglot':
        train_loader, val_loader, test_loader, args = load_omniglot(args, **kwargs)
    elif args.dataset_name == 'caltech101silhouettes':
        train_loader, val_loader, test_loader, args = load_caltech101silhouettes(args, **kwargs)
    elif args.dataset_name == 'histopathologyGray':
        train_loader, val_loader, test_loader, args = load_histopathologyGray(args, **kwargs)
    elif args.dataset_name == 'freyfaces':
        train_loader, val_loader, test_loader, args = load_freyfaces(args, **kwargs)
    elif args.dataset_name == 'cifar10':
        train_loader, val_loader, test_loader, args = load_cifar10(args, **kwargs)
    elif args.dataset_name == 'cifar100':
        train_loader, val_loader, test_loader, args = load_cifar100(args, **kwargs)
    else:
        raise Exception('Wrong name of the dataset!')

    return train_loader, val_loader, test_loader, args
