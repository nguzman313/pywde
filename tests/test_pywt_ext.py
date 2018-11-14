import math
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
from pywde.pywt_ext import Wavelet, WaveletTensorProduct


@pytest.mark.parametrize("wave",[
Wavelet('db1'), Wavelet('db2'), Wavelet('db4'),
Wavelet('db6'), Wavelet('bior1.3')
])
@pytest.mark.parametrize('what', ['base', 'dual'])
@pytest.mark.parametrize('ix', [(0,1,0), (1,1,0), (0,2,1), (1,2,-1)])
def test_fun_support(wave, what, ix):
    "Test values outside support 1-d are zero"
    fun = wave.fun_ix(what, ix)
    a, b = map(np.array, fun.support)
    assert_array_almost_equal(np.zeros(4), fun(np.array([a - 0.5, a - 0.05, b + 0.05, b + 0.5])), 4)


@pytest.mark.parametrize("wave,what,ix,support", [
    [WaveletTensorProduct(('db4', 'db2')),
     'base',
     ((1, 0), (1, 2), (0, 1)),
     (np.array([-3, -0.5]), np.array([4, 1.0]))
     ],
    [WaveletTensorProduct(('db4', 'db2')),
     'dual',
     ((1, 0), (1, 2), (0, 1)),
     (np.array([-3, -0.5]), np.array([4, 1.0]))
     ]
])
def test_fun_support_range(wave, what, ix, support):
    "Test actual support ranges"
    fun = wave.fun_ix(what, ix)
    a, b = map(np.array, fun.support)
    assert_array_almost_equal(support[0], a)
    assert_array_almost_equal(support[1], b)


@pytest.mark.parametrize("wave", [
Wavelet('db1'), Wavelet('db2'), Wavelet('db4'),
Wavelet('db6'), Wavelet('bior1.3')
])
@pytest.mark.parametrize("ix,prec", [
    [(0, 1, 0), 3],
    [(1, 1, 0), 3],
    [(0, 2, 1), 3],
    [(1, 2, -1), 3],
    [(0, 32, -1), 2],
    [(1, 32, -1), 2],
])
def test_norm_1d(wave, ix, prec):
    "Test $\\int \\f_{ix} \\tilde{\\f}_{ix}$ = 1.0 in 1D"
    fun = wave.fun_ix('base', ix)
    fun_dual = wave.fun_ix('dual', ix)
    assert_integral(1, fun, fun_dual, 1.0, prec)


@pytest.mark.parametrize("wave", [
Wavelet('db1'), Wavelet('db2'), Wavelet('db4'),
Wavelet('db6'), Wavelet('bior1.3')
])
@pytest.mark.parametrize("ix,factor,prec", [
    [(1, 1, 0), 2, 3],
    [(1, 2, -1), 4, 3],
    [(1, 32, -1), 2, 2],
])
def test_ortho_case_j_1d(wave, ix, factor, prec):
    fun = wave.fun_ix('base', ix)
    q, s, z = ix
    fun_dual = wave.fun_ix('dual', (q, s * factor, z))
    assert_integral(1, fun, fun_dual, 0.0, prec)


@pytest.mark.parametrize("wave", [
Wavelet('db1'), Wavelet('db2'), Wavelet('db4'),
Wavelet('db6'), Wavelet('bior1.3')
])
@pytest.mark.parametrize("ix,delta_z,prec", [
    [(0, 1, 0), 1, 3],
    [(0, 2, -1), -2, 3],
    [(0, 16, -1), -1, 2],
    [(1, 1, 0), 1, 3],
    [(1, 2, -1), -1, 3],
    [(1, 32, -1), -1, 2],
])
def test_ortho_case_z_1d(wave, ix, delta_z, prec):
    q, s, z = ix
    fun = wave.fun_ix('base', ix)
    fun_dual = wave.fun_ix('dual', (q, s, z - delta_z))
    assert_integral(1, fun, fun_dual, 0.0, prec)


@pytest.mark.parametrize("wave", [
WaveletTensorProduct(('db4', 'db2')),
WaveletTensorProduct(('bior1.3', 'db2'))
])
@pytest.mark.parametrize("ix,prec", [
    [((0,0), (1,2), (0,1)), 2],
    [((0,1), (1,1), (1,1)), 2],
    [((1,0), (2,1), (1,0)), 2],
    [((1,1), (2,2), (0,0)), 2],
    [((1,0), (32,32), (-1,1)), 1],
])
def test_norm_2d(wave, ix, prec):
    "Test $\\int \\f_{ix} \\tilde{\\f}_{ix}$ = 1.0 in 2D"
    fun = wave.fun_ix('base', ix)
    fun_dual = wave.fun_ix('dual', ix)
    assert_integral(2, fun, fun_dual, 1.0, prec)


@pytest.mark.parametrize("wave", [
WaveletTensorProduct(('db2', 'db4')),
WaveletTensorProduct(('db4', 'db4')),
WaveletTensorProduct(('bior1.3', 'db2'))
])
@pytest.mark.parametrize("ix,factors,prec", [
    [((0,1), (1,1), (1,1)), (1,2), 3],
    [((1,0), (2,1), (1,0)), (2,2), 3],
    [((1,1), (2,2), (0,0)), (1,4), 2],
    [((1,0), (32,32), (-1,1)), (2,1), 1],
])
def test_ortho_case_j_2d(wave, ix, factors, prec):
    fun = wave.fun_ix('base', ix)
    qq, ss, zz = ix
    fun_dual = wave.fun_ix('dual', (qq, (ss[0] * factors[0], ss[1] * factors[1]), zz))
    assert_integral(2, fun, fun_dual, 0.0, prec)


@pytest.mark.parametrize("wave", [
WaveletTensorProduct(('db2', 'db4')),
WaveletTensorProduct(('db4', 'db4')),
WaveletTensorProduct(('bior1.3', 'db2'))
])
@pytest.mark.parametrize("ix,delta_z,prec", [
    [((0,0), (1,1), (1,1)), (0,-1), 3],
    [((0,0), (2,1), (1,0)), (-1,0), 3],
    [((0,0), (16,32), (-1,1)), (1,1), 3],
    [((0,1), (1,1), (1,1)), (0,2), 3],
    [((1,0), (2,1), (1,0)), (-2,0), 3],
    [((1,1), (2,2), (0,0)), (1,-1), 3],
    [((1,0), (32,32), (-1,1)), (0,1), 2],
])
def test_ortho_case_z_2d(wave, ix, delta_z, prec):
    fun = wave.fun_ix('base', ix)
    qq, ss, zz = ix
    fun_dual = wave.fun_ix('dual', (qq, ss, (zz[0] - delta_z[0], zz[1] - delta_z[1])))
    assert_integral(2, fun, fun_dual, 0.0, prec)


#
# helpers
#

def assert_integral(dim, f1, f2, exp_value, prec):
    a1, b1 = map(np.array, f1.support)
    a2, b2 = map(np.array, f2.support)
    a = np.amin(np.stack((a1,a2)), axis=0)
    b = np.amax(np.stack((b1,b2)), axis=0)
    if dim == 1:
        val1 = f1(np.linspace(a, b, num=int(10000*(b-a))))
        val2 = f2(np.linspace(a, b, num=int(10000*(b-a))))
        assert_almost_equal(exp_value, (val1 * val2).sum()/10000.0, prec)
    else:
        xx = np.stack((a, b)).T
        x0 = np.linspace(*xx[0], num=int(300*(xx[0,1] - xx[0,0]) + 0.5))
        x1 = np.linspace(*xx[1], num=int(300*(xx[1,1] - xx[1,0]) + 0.5))
        x0, x1 = np.meshgrid(x0, x1)
        val1 = f1((x0, x1))
        val2 = f2((x0, x1))
        assert_almost_equal(exp_value, (val1 * val2).sum()/90000.0, prec)

def assert_almost_equal(float1, float2, prec):
    msg = 'Expected %f, Actual %f (to %d digits)' % (float1, float2, prec)
    fact10 = 10 ** prec
    assert int(math.fabs(float1 - float2) * fact10) == 0, msg
