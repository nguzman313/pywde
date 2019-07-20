import math
import numpy as np

from numpy.testing import assert_array_almost_equal
from pywde.spwde import SPWDE, calc_sqrt_vs, sqrt_vunit


def test_calc_alphas_no_i():
    wave_name = 'db1'
    j0 = 0
    k = 1
    data = np.array([
        [0.5, 0.5],
        [0.6, 0.6],
        [0.7, 0.6],
    ])
    spwde = SPWDE(((wave_name, j0), (wave_name, j0)), k=k)
    balls_info = calc_sqrt_vs(data, k)
    spwde.minx = np.amin(data, axis=0)
    spwde.maxx = np.amax(data, axis=0)
    spwde.calc_funs_at(j0, data)
    alphas = spwde.calc_alphas_no_i(j0, data, 1, balls_info)
    for k, (v1, v2) in alphas.items():
        if k == (0, 0):
            assert math.fabs(v2 - 0.447213595499959) < 0.000001
            assert math.fabs(v1 - 0.447213595499959) < 0.000001
        else:
            assert v1 == 0.0
            assert v2 == 0.0


def test_sqrt_vunit():
    # from wikipedia, volume of n-ball of radious 1
    # https://en.wikipedia.org/wiki/Volume_of_an_n-ball#Low_dimensions
    exp = {1: 2, 2: 3.142, 3: 4.189, 4: 4.935}
    for dim, val in exp.items():
        assert math.fabs(sqrt_vunit(dim) - math.sqrt(val)) < 0.001


def test_calc_sqrt_vs():
    data = np.array([
        [0, 0],
        [1, 1],
        [2, 1],
    ])
    ball_info = calc_sqrt_vs(data, 1)
    assert_array_almost_equal(ball_info.nn_indexes, np.array([[0, 1, 2], [1, 2, 0], [2, 1, 0]]))
    vv = np.array([1.414213562373095, 1, 1]) * sqrt_vunit(2)
    assert_array_almost_equal(ball_info.sqrt_vol_k, vv)
    vv = np.array([2.23606797749979, 1.414213562373095, 2.23606797749979]) * sqrt_vunit(2)
    assert_array_almost_equal(ball_info.sqrt_vol_k_plus_1, vv)

