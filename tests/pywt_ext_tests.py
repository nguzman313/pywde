import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal
from pywde.pywt_ext import Wavelet

import matplotlib.pyplot as plt

class TestWavelet(unittest.TestCase):

    def test_phi_db2(self):
        wave = Wavelet('db2')
        self.assertEqual((0.0, 3.0), wave.phi.support)
        assert_array_almost_equal(np.array([0.0, 1.3660254037844386, -0.3660254037844386, 0.0]), wave.phi(np.array([-0.01, 1.0, 2.0, 3.0])))

    def test_phi_db4(self):
        wave = Wavelet('db4')
        self.assertEqual((0.0, 7.0), wave.phi.support)
        assert_array_almost_equal(np.array([0.0, 1.103055796571806, 0.002152148639222598, 0.0]), wave.phi(np.array([-0.01, 1.1, 4.5, 7.01])))

    def test_psi_db4(self):
        wave = Wavelet('db4')
        self.assertEqual((-3.0, 4.0), wave.psi.support)
        assert_array_almost_equal(np.array([0.0, 1.301367, -0.736155, -0.060386]), wave.psi(np.array([-3.01, 0.66, 1.2, 2.1])))


if __name__ == '__main__':
    unittest.main()
