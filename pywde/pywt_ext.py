"""
Extensions to PyWavelets (pywt) to calculate wavelet values
"""
import math
import re
import pywt
import itertools as itt
import numpy as np
from scipy.interpolate import interp1d


def wavelist():
    return pywt.wavelist()

def np_mult(cols):
    if len(cols) == 1:
        return cols[0]
    if len(cols) == 2:
        return np.multiply(*cols)
    else:
        return np.multiply(cols[0], np_mult(cols[1:]))

def trim_zeros(coeffs):
    nz = np.nonzero(coeffs)
    return coeffs[np.min(nz):np.max(nz) + 1]

def calc_fun(support, values):
    resp = interp1d(np.linspace(*support, num=len(values)), values, fill_value=0.0, bounds_error=False, kind=1)
    resp.support = support
    return resp

_RE = re.compile('(rbio|bior)([0-9]+)[.]([0-9]+)')

def wave_support_info(pywt_wave):
    resp = {}
    if pywt_wave.family_name in ['Daubechies', 'Symlets']:
        phi_support = (0, pywt_wave.dec_len - 1)
        psi_support = (1 - pywt_wave.dec_len // 2, pywt_wave.dec_len // 2)
        resp['base'] = (phi_support, psi_support)
        resp['dual'] = (phi_support, psi_support)
    elif pywt_wave.family_name in ['Coiflets']:
        phi_support = (1 - pywt_wave.dec_len // 2, pywt_wave.dec_len // 2)
        psi_support = (1 - pywt_wave.dec_len // 2, pywt_wave.dec_len // 2)
        resp['base'] = (phi_support, psi_support)
        resp['dual'] = (phi_support, psi_support)
    elif pywt_wave.family_name in ['Biorthogonal', 'Reverse biorthogonal']:
        # pywt uses Spline Wavelets
        # base is the reconstruction family, dual the decomposition family
        m = _RE.match(pywt_wave.name)
        min_vm = int(m.group(2))
        max_vm = int(m.group(3))
        # support for primal lowpass and dual lowpass
        n1, n2 = (-(min_vm // 2) , (min_vm + 1)//2)
        nd1, nd2 = (-max_vm - min_vm // 2 + 1, max_vm + (min_vm - 1) // 2)
        # from this, we can infer support for highpass, so all becomes ...
        resp['base'] = ((n1, n2), ((n1 - nd2 + 1) // 2, (n2 - nd1 + 1) // 2))
        resp['dual'] = ((nd1, nd2), ((nd1 - n2 + 1) // 2, (nd2 - n1 + 1) // 2))
        if pywt_wave.family_name == 'Reverse biorthogonal':
            resp['base'], resp['dual'] = resp['dual'], resp['base']
    else:
        raise ValueError('pywt_wave family %s not known support' % pywt_wave.family_name)
    return resp

def calc_wavefuns(pywt_wave, support):
    values = pywt_wave.wavefun(level=14)
    phi_support_r, psi_support_r = support['base']
    phi_support_d, psi_support_d = support['dual']
    if len(values) == 5:
        phi_d, psi_d, phi_r, psi_r, xx = values
        # biorthogonal wavelets have zeros in pywt for some reason; we have to remove them
        # to match the support
        phi_d, psi_d, phi_r, psi_r = [trim_zeros(c) for c in [phi_d, psi_d, phi_r, psi_r]]
    else:
        phi_d, psi_d, _ = values
        phi_r, psi_r = phi_d, psi_d
    # reconstruction '_r' is the base
    phi = calc_fun(phi_support_r, phi_r)
    psi = calc_fun(psi_support_r, psi_r)
    funs = {}
    funs['base'] = (phi, psi)
    # decomposition '_d' is the dual
    phi = calc_fun(phi_support_d, phi_d)
    psi = calc_fun(psi_support_d, psi_d)
    funs['dual'] = (phi, psi)
    return funs


class Wavelet(pywt.Wavelet):
    """Wrapper around pywt.Wavelet that defines support, phi, psi methods for the base wavelets and
    corresponding duals. If they are orthogonal base and duals are the same. The methods
    work on 1D numpy arrays iterating over elements. For consistency in terminology, use duals to
    calculate coefficients and base to reconstruct the signal (e.g. eq (3.22) in
    1992, Chui, Wang, "On Compactly Supported Spline Wavelets and a Duality Principle")
    """
    def __init__(self, wave_name):
        self.wave = pywt.Wavelet(wave_name)
        self.support = wave_support_info(self.wave)
        self.funs = calc_wavefuns(self.wave, self.support)

    @property
    def phi_prim(self, ix=(1, 0)):
        return self.fun_ix(self.funs['base'][0], ix)

    @property
    def psi_prim(self, ix=(1, 0)):
        return self.fun_ix(self.funs['base'][1], ix)

    @property
    def phi_dual(self, ix=(1, 0)):
        return self.fun_ix(self.funs['dual'][0], ix)

    @property
    def psi_dual(self, ix=(1, 0)):
        return self.fun_ix(self.funs['dual'][1], ix)

    @staticmethod
    def fun_ix(fun, ix):
        """Given fun, returns new function fun(s * x + z), where (s, z) is a parametrisation in ix"""
        s, z = ix
        a, b = fun.support
        f = lambda x: fun(s * x + z)
        f.support = ((a - z)/s, (b - z)/s)
        return f


class WaveletTensorProduct(object):
    """
    Tensor product wavelet in $R^d$. It supports similar or different wavelets in each dimension.

        wave1 = WaveletTensorProduct(('db4',) * 3) # db4 in all 3 axes
        wave2 = WaveletTensorProduct(('rbio2.4', 'rbio1.3', 'rbio3.5')) # three different spline wavelets
    """
    def __init__(self, wave_names, single_j=True):
        self.single_j = single_j
        self.dim = len(wave_names)
        self.waves = [Wavelet(name) for name in wave_names]
        self.qq = list(itt.product(range(2), repeat=self.dim))

    def prim(self, ix=None):
        if ix is None:
            # if ix is None, returns the scaling base at (1,0) in all dimensions
            ix = ((0,) * self.dim, ((1,) * self.dim), ((0,) * self.dim))
        return self.fun_ix('base', ix)

    def dual(self, ix=None):
        if ix is None:
            # if ix is None, returns the scaling dual at (1,0) in all dimensions
            ix = ((0,) * self.dim, ((1,) * self.dim), ((0,) * self.dim))
        return self.fun_ix('dual', ix)

    def fun_ix(self, what, ix):
        qq, ss, zz = ix
        ss2 = math.sqrt(np.prod(ss))
        support = [self.waves[i].support[what][q2] for i, q2 in enumerate(qq)]
        def f(xx):
            cols = []
            for i, q2 in enumerate(qq):
                wave = self.waves[i].funs[what][q2]
                xs_proj = xx[:,i] # proj(xs, i)
                cols.append(wave[q2](ss[i] * xs_proj - zz[i]))
            return np_mult(tuple(cols)) * ss2
        f.dim = self.dim
        f.support = support
        return f
