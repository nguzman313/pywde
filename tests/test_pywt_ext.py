import math
import pytest
import numpy as np
import itertools as itt
from numpy.testing import assert_array_almost_equal
from pywde.common import all_zs_tensor
from pywde.pywt_ext import Wavelet, WaveletTensorProduct

from .conftest import intersect_1d, intersect_2d, assert_integral


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
    "Test actual support ranges in 2D"
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
    "Test $\\psi$ variants are orthogonal on change of scale"
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
    "Test $\\psi$ and $\\phi$ are orthogonal on change of translation"
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
WaveletTensorProduct(('db2', 'db3')),
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
    "Test tensor prods with $\\psi$ variants are orthogonal on change of scale"
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
    "Test tensor prods are orthogonal on change of translation"
    fun = wave.fun_ix('base', ix)
    qq, ss, zz = ix
    fun_dual = wave.fun_ix('dual', (qq, ss, (zz[0] - delta_z[0], zz[1] - delta_z[1])))
    assert_integral(2, fun, fun_dual, 0.0, prec)


@pytest.mark.parametrize("wave", [
Wavelet('db1'), Wavelet('db3'), Wavelet('db4'),
Wavelet('db2'), Wavelet('bior1.3')
])
@pytest.mark.parametrize("what", ['base', 'dual'])
@pytest.mark.parametrize("ix", [
    (0, 1, 0),
    (0, 2, 1),
    (0, 8, -1),
    (1, 1, 0),
    (1, 2, 1),
    (1, 8, -1),
])
def test_z_range_1d(wave, what, ix):
    "Test several facts with the range of z values"
    minx, maxx = 1/3, 2/3
    q, s, z = ix
    z_min, z_max = wave.z_range(what, ix, minx, maxx)
    assert not intersect_1d((minx, maxx), wave.fun_ix(what, (q, s, z_min - 1)).support)
    assert not intersect_1d((minx, maxx), wave.fun_ix(what, (q, s, z_max + 1)).support)
    for zi in range(z_min, z_max+1):
        assert intersect_1d((minx, maxx), wave.fun_ix(what, (q, s, zi)).support)


@pytest.mark.parametrize("wave", [
WaveletTensorProduct(('db2', 'db4')),
WaveletTensorProduct(('db4', 'db4')),
WaveletTensorProduct(('bior1.3', 'db2')),
WaveletTensorProduct(('db2', 'db2')),
])
@pytest.mark.parametrize("what", ['base', 'dual'])
@pytest.mark.parametrize("ix", [
    ((0,0), (1,1), (1,1)),
    ((0,0), (2,1), (1,0)),
    ((0,0), (1,2), (3,0)),
    ((0,1), (1,1), (1,1)),
    ((1,0), (2,1), (1,0)),
    ((1,1), (2,2), (0,0)),
])
def test_z_range_2d(wave, what, ix):
    "Test several facts with the range of z values"
    # region is p1=(1/3,1/4) & p2=(3/4,2/3)
    minx, maxx = np.array((1/3, 1/4)), np.array((3/4, 2/3))
    qq, ss, zz = ix
    zs_min, zs_max = wave.z_range(what, ix, minx, maxx)
    one0 = np.array([1,0])
    one1 = np.array([0,1])
    assert not intersect_2d((minx, maxx), wave.fun_ix(what, (qq, ss, zs_min - one0)).support)
    assert not intersect_2d((minx, maxx), wave.fun_ix(what, (qq, ss, zs_min - one1)).support)
    assert not intersect_2d((minx, maxx), wave.fun_ix(what, (qq, ss, zs_max + one0)).support)
    assert not intersect_2d((minx, maxx), wave.fun_ix(what, (qq, ss, zs_max + one1)).support)
    for zz in itt.product(*all_zs_tensor(zs_min, zs_max)):
        assert intersect_2d((minx, maxx), wave.fun_ix(what, (qq, ss, zz)).support)

def test_z1():
    wave = WaveletTensorProduct(('db1', 'db1'))
    what = 'dual'
    ix = ((0,0), (1,2), (0,0))
    minx, maxx = np.array((0.2, 0.2)), np.array((0.4, 0.6))
    qq, ss, zz = ix
    zs_min, zs_max = wave.z_range(what, ix, minx, maxx)
    print(zs_min, zs_max)
    one0 = np.array([1,0])
    one1 = np.array([0,1])
    assert not intersect_2d((minx, maxx), wave.fun_ix(what, (qq, ss, zs_min - one0)).support)
    assert not intersect_2d((minx, maxx), wave.fun_ix(what, (qq, ss, zs_min - one1)).support)
    assert not intersect_2d((minx, maxx), wave.fun_ix(what, (qq, ss, zs_max + one0)).support)
    assert not intersect_2d((minx, maxx), wave.fun_ix(what, (qq, ss, zs_max + one1)).support)
    for zz in itt.product(*all_zs_tensor(zs_min, zs_max)):
        assert intersect_2d((minx, maxx), wave.fun_ix(what, (qq, ss, zz)).support)
