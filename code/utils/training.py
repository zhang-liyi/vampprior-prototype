from __future__ import print_function

import torch
from torch.autograd import Variable

import numpy as np
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# ======================================================================================================================
def train_vae(epoch, args, train_loader, model, optimizer):
    # set loss to 0
    train_loss = 0
    train_re = 0
    train_kl = 0
    # set model in training mode
    model.train()

    # start training
    if args.warmup == 0:
        beta = 1.
    else:
        beta = 1.* epoch / args.warmup
        if beta > 1.:
            beta = 1.
    print('beta: {}'.format(beta))

    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        # dynamic binarization
        if args.dynamic_binarization:
            x = torch.bernoulli(data)
        else:
            x = data

        # reset gradients
        optimizer.zero_grad()
        # loss evaluation (forward pass)
        loss, RE, KL = model.calculate_loss(x, beta, average=True)
        # backward pass
        loss.backward()
        # optimization
        optimizer.step()

        train_loss += loss.detach().cpu().numpy()
        train_re += -RE.detach().cpu().numpy()
        train_kl += KL.detach().cpu().numpy()

        if args.reduce_datapoints and batch_idx >= 99:
            break

    # calculate final loss
    if not args.reduce_datapoints:
        train_loss /= len(train_loader)  # loss function already averages over batch size
        train_re /= len(train_loader)  # re already averages over batch size
        train_kl /= len(train_loader)  # kl already averages over batch size
    else:
        train_loss /= 20  # loss function already averages over batch size
        train_re /= 20 # re already averages over batch size
        train_kl /= 20 # kl already averages over batch size

    return model, train_loss, train_re, train_kl
