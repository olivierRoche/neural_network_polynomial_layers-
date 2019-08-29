import __init__
import poly_torch
import unittest
import torch
from torch.nn.parameter import Parameter


class CustomTestCase(unittest.TestCase):
    def assertAlmostEqualTensors(self, tensor1, tensor2, rtol=1e-05, atol=1e-08, equal_nan=False):
        if tensor1.shape != tensor2.shape:
            raise AssertionError("Shapes {0} and {1} are not compatible, assertAlmostEqualTensors failed".format(tensor1.shape,
                                                                                                                 tensor2.shape))
        else:
            torch.testing.assert_allclose(tensor1, tensor2, rtol, atol, equal_nan)


class TestForward(CustomTestCase):
    def setUp(self):
        self.poly = poly_torch.PolynomialLayer(degree=2, inp_size=2, out_size=2)
        # we set poly(X,Y) = (X^2 + 2 XY + 2 X + Y + 1, -2 X^2 + 2 XY + Y^2 + 3 X + 2 Y + 2)
        self.poly.coeff_deg_0 = Parameter(torch.tensor([[1.0], [2.0]]))
        self.poly.coeff_deg_1 = Parameter(torch.tensor(([[2.0, 1.0], [3.0, 2.0]])))  # 2 X + Y, 3 X + 2 Y
        self.poly.coeff_deg_2 = Parameter(torch.tensor([[1.0, -2.0, 0.0], [-2.0, 2.0, 1.0]]))  # X^2 - 2 XY + 0 Y^2, -2 X^2 + 2 XY + Y^2

    def test_forward(self):
        batch = torch.tensor([[2.0, 1.0], [3.0, 1.0]])
        self.assertAlmostEqualTensors(self.poly(batch), torch.tensor([[6., 7.], [11., 2.]]))


if __name__ == '__main__':
    unittest.main()