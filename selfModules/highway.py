# -*- coding: utf-8 -*-
import torch as t
import torch.nn as nn
import torch.nn.functional as F


class Highway(nn.Module):
    def __init__(self, size: int, num_layers: int, f: object) -> None:
        """
        [summary] initializes the highway neural network module

        Args:
            size (int): [description] size of the linear layer throughput
            num_layers (int): [description] amount of layers
            f (object): [description] linear layer object
        """
        super(Highway, self).__init__()

        self.num_layers = num_layers

        self.nonlinear = [nn.Linear(size, size) for _ in range(num_layers)]
        for i, module in enumerate(self.nonlinear):
            self._add_to_parameters(module.parameters(), "nonlinear_module_{}".format(i))

        self.linear = [nn.Linear(size, size) for _ in range(num_layers)]
        for i, module in enumerate(self.linear):
            self._add_to_parameters(module.parameters(), "linear_module_{}".format(i))

        self.gate = [nn.Linear(size, size) for _ in range(num_layers)]
        for i, module in enumerate(self.gate):
            self._add_to_parameters(module.parameters(), "gate_module_{}".format(i))

        self.f = f

    def forward(self, x: object) -> object:
        """
        :param x: tensor with shape of [batch_size, size]

        :return: tensor with shape of [batch_size, size]

        applies σ(x) ⨀ (f(G(x))) + (1 - σ(x)) ⨀ (Q(x)) transformation | G and Q is affine transformation,
            f is non-linear transformation, σ(x) is affine transformation with sigmoid non-linearition
            and ⨀ is element-wise multiplication
        """

        for layer in range(self.num_layers):
            gate = t.sigmoid(self.gate[layer](x))
            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)
            x = gate * nonlinear + (1 - gate) * linear

        return x

    def _add_to_parameters(self, parameters: object, name: str) -> None:
        """ add parameters from main RVAE to highway network """
        for i, parameter in enumerate(parameters):
            self.register_parameter(name="{}-{}".format(name, i), param=parameter)
