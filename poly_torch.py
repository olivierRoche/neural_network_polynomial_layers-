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
class Polynomial(Module):
    """ a torch module for a polynomial NN layer.

    Representation of monomials:
        Call a tuple t ordered if t[i] <= t[j] whenever i < j.
        A monomial in the variables X_0, ..., X_n of degree d is represented by an ordered tuple of length d which
        values range in 0, ..., n. The degree of the variable X_i is the number of occurences of i in this tuple. eg:
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
            the coefficients of the monomials of degree d for each polynomial P_{0}, ..., P_{out_size - 1}. These
            coefficients are enumerated in lexycographical order. eg, say inp_size == 2, theb coeff_deg_2[1] describes
            the coefficients of degree 2 of P_1 as follow:
                * coeff_deg_2[1][0] is the coefficient of X_0 * X_0
                * coeff_deg_2[1][1] is the coefficient of X_0 * X_1
                * coeff_deg_2[1][2] is the coefficient of X_0 * X_2
                * coeff_deg_2[1][3] is the coefficient of X_1 * X_1
                * coeff_deg_2[1][4] is the coefficient of X_1 * X_2
                * coeff_deg_2[1][5] is the coefficient of X_2 * X_2
    """
    def __init__(self, degree, inp_size, out_size):
        super(Polynomial, self).__init__()
        self.degree = degree
        self.inp_size = inp_size
        self.out_size = out_size
        self.parameter_names = ["coeff_deg_{0}".format(d) for d in range(degree + 1)]
        self.param_size = self.__count_ordered_tuples(0, degree)
        """ We use reflexion to set the Parameters as attribute. This way, they are automatically added to the list of
            the parameters of the Module Polynomials (cf the doc of Parameter).
        """
        for d in range(degree + 1):
            setattr(self, "coeff_deg_{0}".format(d),
                    Parameter(torch.Tensor(out_size, self.__count_ordered_tuples(0, d))))
        self.reset_parameters()
        # self.tuple_indices stores the index of each ordered_tuple for easy access in various tensors
        self.tuple_indices = {t: n for d in range(1, degree + 1)
                              for n, t in enumerate(iter_ordered_tuples(0, inp_size, d))}

    def __count_ordered_tuples(self, start, deg):
        if deg <= 0:
            return 1
        if deg == 1:
            return self.inp_size - start
        else:
            return sum(self.__count_ordered_tuples(i, deg - 1) for i in range(start, self.inp_size))

    def reset_parameters(self):
        for param in self.parameter_names:
            init.kaiming_uniform_(getattr(self, param), a=sqrt(5))

    def _autopad(self, input):
        """ automaticaly pads the forelast axis of tensor input so that this axis becomes of size self.param_size
        Usage examples:
            turns self.coeff_deg_d into a tensor of shape (self.param_size, self.out_size)
        """
        return F.pad(input=input, pad=[0, self.param_size - input.shape[-1]])

    @weak_script_method
    def forward(self, input):
        monomial_rows = [self._autopad(torch.tensor([1.0])), self._autopad(input)]
        """ monomial_rows[d] will contain the values of the monomials of degree d for the given input.
        Using monomial_rows[d] to compute monomial_rows[d + 1] allows us a memoization.
        """
        for d in range(2, self.degree + 1):
            degree_d = torch.tensor([input[t[0]] * monomial_rows[-1][self.tuple_indices[t[1:]]] for t in
                                     iter_ordered_tuples(0, len(input), d)])
            monomial_rows.append(self._autopad(degree_d))
        return sum(self._autopad(getattr(self, "coeff_deg_{0}".format(d))) @ monomial_rows[d] for d in range(self.degree + 1))


if __name__ == "__main__":
    p = Polynomial(degree=3, inp_size=3, out_size=4)
    print(p.coeff_deg_0)
    for t in iter_ordered_tuples(0, 3, 2):
        print(t)
    print(p(torch.tensor([2.0, 3.0, 5.0])))
