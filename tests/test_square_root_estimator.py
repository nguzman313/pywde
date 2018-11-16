import math
import pytest
import mock
import numpy as np
from numpy.testing import assert_array_almost_equal
from pywde.square_root_estimator import WParams, WaveletDensityEstimator, WaveletTensorProduct
from .conftest import intersect_2d, assert_almost_equal


@pytest.fixture
def mock_wde_1():
    obj = mock.Mock()
    obj.k = 1
    obj.wave = WaveletTensorProduct(('db2', 'db1'))
    obj.jj0 = np.array([0, 1])
    obj.delta_j = 1
    return obj

@pytest.fixture
def mock_wde_1_with_data(mock_wde_1):
    data = np.array([
        (0.21, 0.21),
        (0.21, 0.51),
        (0.41, 0.61),
    ])
    mock_wde_1.minx = np.array([0.2, 0.2])
    mock_wde_1.maxx = np.array([0.4, 0.6])
    return mock_wde_1, data


def test_wparam_calc_indexes_j_qq(mock_wde_1_with_data):
    wde, data = mock_wde_1_with_data
    wparams = WParams(wde)
    ixs = set()
    for tup in wparams.coeffs.keys():
        j, qq, zz, jpow2 = tup
        assert j in [0,1]
        assert jpow2 == tuple(np.array([1,2]) * (2 ** j))
        assert qq in [(0,0),(0,1),(1,0),(1,1)]
        if qq == (0,0):
            assert j == 0
        if j > 0:
            assert qq != (0,0)
        assert (j, qq, zz) not in ixs
        ixs.add((j, qq, zz))


def test_wparam_calc_indexes_zz(mock_wde_1_with_data):
    wde, data = mock_wde_1_with_data
    wparams = WParams(wde)
    for tup in wparams.coeffs.keys():
        j, qq, zz, jpow2 = tup
        jpow2 = np.array(jpow2)
        fun = wde.wave.fun_ix('dual', (qq, jpow2, zz))
        assert intersect_2d((wde.minx, wde.maxx), fun.support)


def test_balls(mock_wde_1_with_data):
    wde, data = mock_wde_1_with_data
    wparams = WParams(wde)
    wparams.calc_coeffs(data)
    assert_array_almost_equal(np.array([0.531736,0.396333,0.396333]), wparams.xs_balls)


def test_betas(mock_wde_1_with_data):
    wde, data = mock_wde_1_with_data
    wparams = WParams(wde)
    wparams.calc_coeffs(data)
    coeff, num = wparams.coeffs[(0, (1,1), (0,0), (1,2))]
    assert_almost_equal(0.452659, coeff, 3)
    assert num == 1
    coeff, num = wparams.coeffs[(0, (1,1), (0,-1), (1,2))]
    assert_almost_equal(0.871158, coeff, 3)
    assert num == 2


def _test_wparam_calc_coeffs_no_cv(mock_wde_with_data):
    wde, data = mock_wde_with_data
    wparams = WParams(wde)
    wparams.calc_coeffs(data)
    assert len(wparams.coeffs.keys()) == 0
