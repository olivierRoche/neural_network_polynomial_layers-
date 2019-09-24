#!/usr/bin/env python
""" A pytorch Module for polynomial (rather than affine) layers. """


import torch
import torch.nn as nn
import numpy as np
from torch.nn.modules import Module
from torch._jit_internal import weak_module, weak_script_method
from torch.nn.parameter import Parameter
from torch.nn.modules.linear import init
from math import sqrt
import torch.nn.functional as F
import torch.optim
from tqdm import tqdm

__author__ = "Olivier Roche"
__copyright__ = """This file is part of neural_network_polynomial_layers.

    neural_network_polynomial_layers is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    neural_network_polynomial_layers is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with neural_network_polynomial_layers.  If not, see <https://www.gnu.org/licenses/>.
    """
__credits__ = ["Olivier Roche", "Thibault Ketterer"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Olivier Roche"
__email__ = "olivier.a.roche@gmail.com"
__status__ = "Prototype"

def iter_ordered_tuples(start, stop, length):
    """ enumerates all ordered tuples of given length whose elements are in range(start, stop) lexycographically

    Parameters:
        start : int
        stop : int
        length : int

    Yields:
        ordered tuples of ints of given length
    """
    if length == 0:
        yield ()
    else:
        for fst in range(start, stop):
            for t in iter_ordered_tuples(fst, stop, length - 1):
                yield (fst,) + t


@weak_module
class PolynomialLayer(Module):
    """ a torch module for a polynomial NN layer.

    Representation of monomials:
        Call a tuple t ordered if t[i] <= t[j] whenever i < j.
        A monomial in the variables X_0, ..., X_n of degree d is represented by an ordered tuple of length d which
        values range in 0, ..., n. The degree of the variable X_i is the number of occurrences of i in this tuple. eg:
            * (0, 1, 2) represents X_0 * X_1 * X_2.
            * (0, 0, 2) represents X_0 * X_0 * X_2, ie X_0^2 * X_2
            * (0, 4, 4, 4) represents X_0 * X_4^3

    Attributes:
        degree : int
            the degree of each polynomial
        inp_size : int
            the number of variables of each polynomial
        out_size : int
            the size of the output. Hence, there are out_size polynomials, say P_{0}, ..., P_{out_size - 1}. The i-th
            component of the output is given by P_i(input)
        param_size : int
            the maximum size for the last axis of the Parameters self.coeff_deg_d
        coeff_deg_d : torch.nn.parameter.Parameter
            the coefficients of the monomials of degree d for each PolynomialLayer P_{0}, ..., P_{out_size - 1}. These
            coefficients are enumerated in lexycographical order. eg, say inp_size == 2, then coeff_deg_2[1] describes
            the coefficients of degree 2 of P_1 as follow:
                * coeff_deg_2[1][0] is the coefficient of X_0 * X_0
                * coeff_deg_2[1][1] is the coefficient of X_0 * X_1
                * coeff_deg_2[1][2] is the coefficient of X_0 * X_2
                * coeff_deg_2[1][3] is the coefficient of X_1 * X_1
                * coeff_deg_2[1][4] is the coefficient of X_1 * X_2
                * coeff_deg_2[1][5] is the coefficient of X_2 * X_2
        place_tensor : int tensor of shape (self.degree, self.inp_size)
            place_tensor[k, d] contains the number of monomials in k variables of degree d.
    """
    def __init__(self, degree, inp_size, out_size):
        super(PolynomialLayer, self).__init__()
        self.degree = degree
        self.inp_size = inp_size
        self.out_size = out_size
        self.param_size = self.count_ordered_tuples(0, degree)
        """ We use reflexion to set the Parameters as attribute. This way, they are automatically added to the list of
            the parameters of the Module PolynomialLayer (cf the doc of Parameter).
        """
        for d in range(degree + 1):
            setattr(self, "coeff_deg_{0}".format(d),
                    Parameter(torch.Tensor(out_size, self.count_ordered_tuples(0, d))))
        self.reset_parameters()
        # self.tuple_indices stores the index of each ordered_tuple for easy access in various tensors
        self.tuple_indices = {t: n for d in range(1, degree + 1)
                              for n, t in enumerate(iter_ordered_tuples(0, inp_size, d))}
        self.place_tensor, self.rplace_tensor = self.compute_place_tensor()

    def count_ordered_tuples(self, start, deg):
        """ returns the number of ordered tuples of length deg with values in range(start, self.inp_size) """
        if deg <= 0:
            return 1
        if deg == 1:
            return self.inp_size - start
        else:
            return sum(self.count_ordered_tuples(i, deg - 1) for i in range(start, self.inp_size))

    def reset_parameters(self):
        for param in self.parameters():
            init.kaiming_uniform_(param, a=sqrt(5))

    def compute_place_tensor(self):
        """ returns place_tensor and rplace_tensor, two int tensors of shape (self.degree, self.inp_size)
        place_tensor(d,i) is the number of monomials of degree d+1 begining with X_i
        rplace_tensor(d,i) is the number of monomials of degree self.degree - (d+1) begining with X_i """
        self.place_tensor = torch.ones(self.degree , self.inp_size, dtype=torch.int64, requires_grad=False)
        for d in range(1, self.degree):
            for k in range(1, self.inp_size):
                self.place_tensor[d, k] = torch.sum(self.place_tensor[d - 1][:k + 1])
        self.rplace_tensor = torch.from_numpy(np.flip(self.place_tensor.numpy(), 0).copy())
        return self.place_tensor, self.rplace_tensor

    @weak_script_method
    def forward(self, inp):
        inp = inp.view(-1, self.inp_size)
        batch_size = inp.shape[0]
        monomial_rows = [torch.ones(batch_size, self.out_size), inp]
        """ monomial_rows[d] will contain the values of the monomials of degree d for the given input.
        Using monomial_rows[d] to compute monomial_rows[d + 1] allows us a memoization.
        """
        for d in range(2, self.degree + 1):
            degree_d = torch.tensor([[inp[b, t[0]] * monomial_rows[-1][b, self.tuple_indices[t[1:]]] for t in
                                      iter_ordered_tuples(0, self.inp_size, d)] for b in range(batch_size)])
            monomial_rows.append(degree_d)
        ret = sum([torch.matmul(getattr(self, "coeff_deg_{0}".format(d)),
                                         torch.t(monomial_rows[d])) for d in range(1, self.degree + 1)])
        return ((ret + torch.cat([self.coeff_deg_0 for i in range(batch_size)], dim=-1)).view(batch_size, self.out_size)).t()
        # return torch.t(sum([torch.matmul(getattr(self, "coeff_deg_{0}".format(d)),
        #                                  torch.t(monomial_rows[d])) for d in range(1, self.degree + 1)]) +
        #                torch.cat([self.coeff_deg_0 for i in range(batch_size)], dim=-1))
        # return sum(self._autopad(getattr(self, "coeff_deg_{0}".format(d))) @ monomial_rows[d] for d in range(self.degree + 1))

    def norm_coefficients(self, coefficients):
        """ returns the maximum of the norm of each polynomial described by coefficients
        Parameters:
             coefficients : list of tensors
                a list of tensors describing polynomials the same way that self does, ie:
                -- len(coefficients) ==  self.degree
                -- coefficients[d].shape == self.coeff_deg_d.shape for all d <= self.degree

        Returns:
            max_norm : float
        """
        with torch.no_grad():
            concatenation = torch.cat(coefficients, -1)
            norms = torch.norm(concatenation, dim=-1)
            max_norm = torch.max(norms)
        return max_norm.item()

    def fit_to_training(self, training, optimizer, lr=0.1, normalize=True):
        """ fits the (torch) neural network self to the data contained in training, with a progression bar

        Parameters:
            training : list of pairs (input : tensor, expected : tensor)
                the data used to train the nn
            optimizer : an optimizer from torch.optim, eg torch.optim.Adam or torch.optim.SGD
            lr : float
                the learning rate
            normalize : bool
                if True, normalizes each learning step. This option is useful to prevent overflow during training.
        """
        self.train()
        for batch_idx, (data, target) in tqdm(enumerate(training), total=len(training)):
            optim = optimizer(self.parameters(), lr=lr)
            optim.zero_grad()
            output = self(data)
            loss = torch.nn.MSELoss()
            error = loss(output, target)
            error.backward(retain_graph=True)
            if normalize:
                with torch.no_grad():
                    norm = self.norm_coefficients([param.grad.data for param in self.parameters()])
                if norm > 1:
                    for param in self.parameters():
                        param.grad.data /= norm
            optim.step()

if __name__ == "__main__":
    target_polynomial = PolynomialLayer(degree=2, inp_size=1, out_size=1)
    target_polynomial.coeff_deg_0 = Parameter(torch.tensor([1.0]))
    target_polynomial.coeff_deg_1 = Parameter(torch.tensor([2.0]))
    target_polynomial.coeff_deg_2 = Parameter(torch.tensor([1.0]))
    training = []

    for i in range(10000):
        v = 10 * torch.rand(2,1)
        training.append((v, target_polynomial(v) + torch.rand(1) - 0.5))
    poly = PolynomialLayer(degree=4, inp_size=1, out_size=1)
    poly.fit_to_training(training, torch.optim.SGD)
    print(poly.place_tensor)
    print(poly.rplace_tensor)