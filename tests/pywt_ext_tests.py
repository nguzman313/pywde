import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal
from pywde.pywt_ext import Wavelet


class WaveletCase(object):

    def test_fun_support(self):
        a, b = self.fun.support
        assert_array_almost_equal(np.zeros(4), self.fun(np.array([a - 0.5, a - 0.05, b + 0.05, b + 0.5])), 4)

    def test_dual_support(self):
        a, b = self.fun_dual.support
        assert_array_almost_equal(np.zeros(4), self.fun_dual(np.array([a - 0.5, a - 0.05, b + 0.05, b + 0.5])), 4)

    def test_fun_dual_integral(self):
        a1, b1 = self.fun.support
        a2, b2 = self.fun_dual.support
        a = min(a1,a2)
        b = max(b1,b2)
        val1 = self.fun(np.linspace(a, b, num=int(10000*(b-a))))
        val2 = self.fun_dual(np.linspace(a, b, num=int(10000*(b-a))))
        self.assertAlmostEqual(1.0, (val1 * val2).sum()/10000.0, 4)


class TestDb2Phi(unittest.TestCase, WaveletCase):
    def setUp(self):
        wave = Wavelet('db2')
        self.fun = wave.phi_prim
        self.fun_dual = wave.phi_dual

    def test_phi_prim_values(self):
        assert_array_almost_equal(np.array([1.3660254037844386, -0.3660254037844386]),
                                  self.fun(np.array([1.0, 2.0])), 4)


class TestDb2Psi(unittest.TestCase, WaveletCase):
    def setUp(self):
        wave = Wavelet('db2')
        self.fun = wave.psi_prim
        self.fun_dual = wave.psi_dual


class TestDb4Psi(unittest.TestCase, WaveletCase):
    def setUp(self):
        wave = Wavelet('db4')
        self.fun = wave.psi_prim
        self.fun_dual = wave.psi_dual

    def test_fun_values(self):
        assert_array_almost_equal(np.array([0.0, 1.301367, -0.736155, -0.060386]),
                                  self.fun(np.array([-3.01, 0.66, 1.2, 2.1])), 4)


class TestBior13PhiPrim(unittest.TestCase, WaveletCase):
    def setUp(self):
        wave = Wavelet('bior1.3')
        self.fun = wave.phi_prim
        self.fun_dual = wave.phi_dual


class TestBior13PsiPrim(unittest.TestCase, WaveletCase):
    def setUp(self):
        wave = Wavelet('bior1.3')
        self.fun = wave.psi_prim
        self.fun_dual = wave.psi_dual


if __name__ == '__main__':
    unittest.main()
