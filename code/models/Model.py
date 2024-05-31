from __future__ import print_function

import numpy as np

import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable

from utils.nn import normal_init, NonLinear
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

#=======================================================================================================================
class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.args = args

    # AUXILIARY METHODS
    def add_pseudoinputs(self):

        nonlinearity = nn.Hardtanh(min_val=0.0, max_val=1.0)
        out_dim = np.prod(self.args.input_size) if not self.args.collapse_channels else np.prod(self.args.input_size[1:])

        self.means = NonLinear(self.args.number_components, out_dim, bias=False, activation=nonlinearity)

        # init pseudo-inputs
        if self.args.use_training_data_init:
            self.means.linear.weight.data = self.args.pseudoinputs_mean
        else:
            normal_init(self.means.linear, self.args.pseudoinputs_mean, self.args.pseudoinputs_std)

        # create an idle input for calling pseudo-inputs
        self.idle_input = Variable(torch.eye(self.args.number_components, self.args.number_components), requires_grad=False)
        if self.args.cuda:
            self.idle_input = self.idle_input.cuda()

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.args.cuda:
            eps = torch.zeros(std.size(), dtype=torch.float32, device='cuda').normal_()
        else:
            eps = torch.zeros(std.size(), dtype=torch.float32, device='cpu').normal_()
        eps = Variable(eps)
        return mu + torch.multiply(std, eps)

    def calculate_loss(self):
        return 0.

    def calculate_likelihood(self):
        return 0.

    def calculate_lower_bound(self):
        return 0.

    # THE MODEL: FORWARD PASS
    def forward(self, x):
        return 0.

#=======================================================================================================================