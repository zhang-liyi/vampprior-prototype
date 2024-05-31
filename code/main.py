import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torchvision
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from analytic_utils.load_data import *
from analytic_utils.plot import *
from classifier import CNN
from parse_args import *
from methods import *


args = parse_args()
config = vars(args)
seed = args.seed
K = args.K

def model_generate(model, dataset='mnist', N=500):

    if dataset == 'mnist':
        means = model.means(model.idle_input[:N]).view(-1, model.args.input_size[0], model.args.input_size[1],model.args.input_size[2])
        z2_sample_gen_mean, z2_sample_gen_logvar = model.q_z2(means)
        z2_sample_rand = model.reparameterize(z2_sample_gen_mean, z2_sample_gen_logvar)

        # Sampling z1 from a model
        z1_sample_mean, z1_sample_logvar = model.p_z1(z2_sample_rand)
        z1_sample_rand = model.reparameterize(z1_sample_mean, z1_sample_logvar)

        # Sampling from PixelCNN
        samples_gen = model.pixelcnn_generate(z1_sample_rand, z2_sample_rand)

        return samples_gen
    
    elif dataset == 'cifar10':
        samples_gen = model.generate_x(N)

        return samples_gen

if args.dataset == 'mnist':

    train_loader, val_loader, test_loader = load_static_mnist(256)

    cls_model = CNN((1, 28, 28), 10)
    optimizer = torch.optim.Adam(cls_model.parameters(), 0.001)
    criterion = torch.nn.CrossEntropyLoss()  # loss function

    # Train classifier
    best_acc = 0
    for epoch in range(10):
        train_loss = []
        for i, (data, labels) in enumerate(train_loader):
            # pass data through network
            data = torch.reshape(data, (-1,1,28,28))
            outputs = cls_model(data)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        test_loss = []
        test_accuracy = []
        for i, (data, labels) in enumerate(val_loader):
            # pass data through network
            data = torch.reshape(data, (-1,1,28,28))
            outputs = cls_model(data)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            test_loss.append(loss.item())
            test_accuracy.append((predicted == torch.argmax(labels, 1)).sum().item() / predicted.size(0))
        print('epoch: {}, train loss: {}, test loss: {}, test accuracy: {}'.format(epoch, np.mean(train_loss), np.mean(test_loss), np.mean(test_accuracy)))

        if np.mean(test_accuracy) > best_acc:
            best_acc = np.mean(test_accuracy)
            torch.save(cls_model, 'snapshots/cls_model.model')

    # Load VAE

    dir = 'snapshots/2024-01-28 11_43_26_static_mnist_pixelhvae_2level_vampprior(K_500)_wu(100)_z1_40_z2_40/'
    model = torch.load(f'{dir}pixelhvae_2level.model')
    model_args = torch.load(f'{dir}pixelhvae_2level.config')
    cls_model = torch.load('snapshots/cls_model.model')
    train_loader, val_loader, test_loader = load_static_mnist(1)

    pseudoinputs = model.means(model.idle_input).cpu().data.numpy()
    print(pseudoinputs.shape)
    generations = model_generate(model, args.dataset)
    print('Plotting generations')
    plot_images(model_args, generations.data.cpu().numpy()[:25], 'mnist_generations.pdf', 5,5)
    print('Classifier results on the generated images', torch.argmax(cls_model(generations[:25].cpu()), 1))
    print('Plotting the corresponding pseudoinputs')
    plot_images(model_args, pseudoinputs[:25], 'pseudoinputs.pdf', 5, 5)

    knn_model = run_knn(model, args.dataset, train_loader, val_loader, val_loader)

    _ = run_tsne(model, pseudoinputs, val_loader, model_type=args.model_type)

    scores = run_kmeans(model, val_loader)

elif args.dataset == 'cifar10':

    train_loader, val_loader, test_loader = load_cifar10(1, args.rm_labels)

    dir = 'snapshots/2024-02-01 11_26_09_cifar10_dcganvae_vampprior(K_500)_wu(0)_z1_200_z2_40_noise_FalseFalse0.2_4/'
    model = torch.load(f'{dir}dcganvae.model')
    model_args = torch.load(f'{dir}dcganvae.config')

    pseudoinputs = model.means(model.idle_input).cpu().data.numpy()

    generations = model_generate(model, args.dataset, 25)
    plot_images(model_args, generations.data.cpu().numpy()[:25], 'cifar10_generations.pdf', 5,5)

    knn_model = run_knn(model, args.dataset, train_loader, val_loader, test_loader)

