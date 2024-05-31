import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt



def run_knn(model, dataset, train_loader, val_loader, test_loader):
    
    if dataset == 'mnist':
        z2_q_mean_train = torch.zeros((len(train_loader), 40))
        y_train = torch.zeros((len(train_loader),1))
        z2_q_mean_test = torch.zeros((len(test_loader), 40))
        y_test = torch.zeros((len(test_loader),1))

        for i, (data, labels) in enumerate(train_loader):
            # pass data through network
            data = torch.reshape(data, (-1,1,28,28)).to(torch.device('cuda'))
            z2_q_mean, z2_q_logvar = model.q_z2(data)
            z2_q_mean_train[i,:] = z2_q_mean.detach().cpu()
            y_train[i] = torch.argmax(labels)
            if i % 10000 == 0:
                print(i)

        for i, (data, labels) in enumerate(test_loader):
            # pass data through network
            data = torch.reshape(data, (-1,1,28,28)).to(torch.device('cuda'))
            z2_q_mean, z2_q_logvar = model.q_z2(data)
            z2_q_mean_test[i,:] = z2_q_mean.detach().cpu()
            y_test[i] = torch.argmax(labels)

        neigh = KNeighborsClassifier(n_neighbors=5)
        neigh.fit(z2_q_mean_train[:50000].numpy(), np.squeeze(y_train[:50000].numpy()))

        print('train data accuracy by KNN')
        preds = neigh.predict(z2_q_mean_train[50000:].numpy())
        print(np.mean(preds == np.squeeze(y_train[50000:].numpy())))

        print('val data accuracy by KNN')
        preds = neigh.predict(z2_q_mean_test.numpy())
        print(np.mean(preds == np.squeeze(y_test.numpy())))

    elif dataset == 'cifar10':

        z_q_mean_train = torch.zeros((len(train_loader), 200))
        y_train = torch.zeros((len(train_loader),1))
        z_q_mean_val = torch.zeros((len(val_loader), 200))
        y_val = torch.zeros((len(val_loader),1))
        z_q_mean_test = torch.zeros((len(test_loader), 200))
        y_test = torch.zeros((len(test_loader),1))

        for i, (data, labels) in enumerate(train_loader):
            # pass data through network
            data = torch.reshape(data, (-1,3,32,32)).to(torch.device('cuda'))
            z_q_mean, z_q_logvar = model.q_z(data)
            z_q_mean_train[i,:] = z_q_mean.detach().cpu()
            y_train[i] = labels
            if i % 10000 == 0:
                print(i)

        for i, (data, labels) in enumerate(val_loader):
            # pass data through network
            data = torch.reshape(data, (-1,3,32,32)).to(torch.device('cuda'))
            z_q_mean, z_q_logvar = model.q_z(data)
            z_q_mean_val[i,:] = z_q_mean.detach().cpu()
            y_val[i] = labels

        for i, (data, labels) in enumerate(test_loader):
            # pass data through network
            data = torch.reshape(data, (-1,3,32,32)).to(torch.device('cuda'))
            z_q_mean, z_q_logvar = model.q_z(data)
            z_q_mean_test[i,:] = z_q_mean.detach().cpu()
            y_test[i] = labels

        neigh = KNeighborsClassifier(n_neighbors=12)
        neigh.fit(z_q_mean_train.numpy(), np.squeeze(y_train.numpy()))

        preds = neigh.predict(z_q_mean_val.numpy())
        print(np.mean(preds == np.squeeze(y_val.numpy())))
        preds = neigh.predict(z_q_mean_test.numpy())
        print(np.mean(preds == np.squeeze(y_test.numpy())))
    
    return neigh



def run_tsne(model, pseudoinputs, test_loader, model_type='vampprior'):

    z2_q_mean_all = torch.zeros((len(test_loader), 40))
    z2_q_pseudo_mean_all = torch.zeros((500, 40))
    y = {}
    for i in range(10):
        y[i] = []

    for i, (data, labels) in enumerate(test_loader):
        # pass data through network
        data = torch.reshape(data, (-1,1,28,28)).to(torch.device('cuda'))
        z2_q_mean, z2_q_logvar = model.q_z2(data)
        z2_q_mean_all[i,:] = z2_q_mean
        y[int(torch.argmax(labels).numpy())].append(i)

    K = len(pseudoinputs)
    pseudoinputs_cp = torch.tensor(pseudoinputs).to(torch.device('cuda'))
    pseudoinputs_cp = torch.reshape(pseudoinputs_cp, (-1,1,28,28))
    for i, pinput in enumerate(pseudoinputs_cp):
        z2_q_pseudo_mean, z2_q_pseudo_logvar = model.q_z2(torch.unsqueeze(pinput, dim=0))
        z2_q_pseudo_mean_all[i,:] = z2_q_pseudo_mean

    if model_type == 'vampprior':
        tsne_input = np.concatenate([z2_q_mean_all.detach().cpu().numpy(), z2_q_pseudo_mean_all.detach().cpu().numpy()],
                                0)
    else:
        tsne_input = z2_q_mean_all.detach().cpu().numpy()

    x_embedded = TSNE(perplexity=30, n_iter=500).fit_transform(tsne_input)

    plt.rcParams['figure.figsize'] = [10,10]
    for i in range(10):
        plt.scatter(x_embedded[y[i],0],x_embedded[y[i],1], label=str(i), s=8)
    plt.scatter(x_embedded[-K:,0],x_embedded[-K:,1], label='pseudo-input', s=20, color='black', marker='s')
    plt.legend()
    plt.savefig('tsne' + '.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

    return



def run_kmeans(model, test_loader):
     
    z2_q_mean_all = torch.zeros((len(test_loader), 40))

    for i, (data, labels) in enumerate(test_loader):
        # pass data through network
        data = torch.reshape(data, (-1,1,28,28)).to(torch.device('cuda'))
        z2_q_mean, z2_q_logvar = model.q_z2(data)
        z2_q_mean_all[i,:] = z2_q_mean

    kmeans_input = z2_q_mean_all.detach().cpu().numpy()

    scaler = StandardScaler()
    kmeans_input = scaler.fit_transform(kmeans_input)

    scores = []

    for k in [2,4,6,8,10,12,14,16,18,20,25,30,35,40]:

        kmeans_model = KMeans(k, max_iter=3000)

        kmeans_model.fit(kmeans_input)
        scores.append(kmeans_model.score(kmeans_input))

    print(scores)

    return scores