from __init__ import hack_torch
import unittest
import torch
from torch.nn.parameter import Parameter


class TestForward(unittest.TestCase):
    def setUp(self):
        self.poly = hack_torch.Polynomial(degree=2, inp_size=2, out_size=1)
        # we set poly(X,Y) = (X^2 + 2 XY + 2 X + Y + 1, -2 X^2 + 2 XY + Y^2 + 3 X + 2 Y + 2)
        self.poly.coeff_deg_0 = Parameter(torch.tensor([[1.0], [2.0]]))
        self.poly.coeff_deg_1 = Parameter(torch.tensor(([[2.0, 1.0], [3.0, 2.0]])))  # 2 X + Y, 3 X + 2 Y
        self.poly.coeff_deg_2 = Parameter(torch.tensor([[1.0, -2.0, 0.0], [-2.0, 2.0, 1.0]]))  # X^2 - 2 XY + 0 Y^2, -2 X^2 + 2 XY + Y^2

    def test_forward(self):
        self.assertAlmostEqual(self.poly(torch.tensor([2.0, 1.0]))[0].item(), 6.0)  # P_0(2, 1) == 6
        self.assertAlmostEqual(self.poly(torch.tensor([2.0, 1.0]))[1].item(), 7.0)  # P_1(2, 1) == 7


if __name__ == '__main__':
    unittest.main()