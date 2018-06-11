import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal
from pywde.pywt_ext import Wavelet


class WaveletCase(object):

    def test_phi_prim(self):
        self.assert_support(self.wave.phi_prim)
        self.assert_norm(self.wave.phi_prim)

    def test_psi_prim(self):
        self.assert_support(self.wave.psi_prim)
        self.assert_norm(self.wave.phi_prim)

    def test_phi_dual(self):
        self.assert_support(self.wave.phi_dual)
        self.assert_norm(self.wave.phi_prim)

    def test_psi_dual(self):
        self.assert_support(self.wave.psi_dual)
        self.assert_norm(self.wave.phi_prim)

    @staticmethod
    def assert_support(fun):
        a, b = fun.support
        assert_array_almost_equal(np.zeros(6), fun(np.array([a - 0.5, a - 0.1, a, b, b + 0.1, b + 0.5])), 4)

    def assert_norm(self, fun):
        a, b = fun.support
        val_a = fun(np.linspace(a, b, num=int(10000*(b-a))))
        self.assertAlmostEqual(1.0, (val_a * val_a).sum()/10000.0, 4)


class TestDb2(unittest.TestCase, WaveletCase):
    def setUp(self):
        self.wave = Wavelet('db2')

    def test_phi_prim_values(self):
        assert_array_almost_equal(np.array([1.3660254037844386, -0.3660254037844386]),
                                  self.wave.phi_prim(np.array([1.0, 2.0])), 4)


class TestDb4(unittest.TestCase, WaveletCase):
    def setUp(self):
        self.wave = Wavelet('db4')

    def test_phi_prim_values(self):
        assert_array_almost_equal(np.array([0.0, 1.301367, -0.736155, -0.060386]),
                                  self.wave.psi_prim(np.array([-3.01, 0.66, 1.2, 2.1])), 4)

    def test_psi_prim_values(self):
        assert_array_almost_equal(np.array([0.0, 1.301367, -0.736155, -0.060386]),
                                  self.wave.psi_prim(np.array([-3.01, 0.66, 1.2, 2.1])), 4)


class TestBior24(unittest.TestCase, WaveletCase):
    def setUp(self):
        self.wave = Wavelet('bior2.4')


if __name__ == '__main__':
    unittest.main()
