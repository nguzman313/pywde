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

def trim_zeros(coeffs):
    nz = np.nonzero(coeffs)
    return coeffs[np.min(nz):np.max(nz) + 1]

def calc_fun(support, values):
    resp = interp1d(np.linspace(*support, num=len(values)), values, fill_value=0.0, bounds_error=False, kind=2)
    resp.support = support
    return resp

_RE1 = re.compile('(db|sym)([0-9]+)')
_RE = re.compile('(rbio|bior)([0-9]+)[.]([0-9]+)')
_RESOLUTION_1D = 14

def wave_support_info(pywt_wave):
    resp = {}
    if pywt_wave.family_name in ['Daubechies', 'Symlets']:
        match = _RE1.match(pywt_wave.name)
        vm = int(match.group(2))
        phi_support = (0, 2 * vm - 1)
        psi_support = (1 - vm, vm)
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
    values = pywt_wave.wavefun(level=_RESOLUTION_1D)
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
        self.dim = 1

    @property
    def phi_prim(self, ix=(1, 0)):
        return self.fun_ix('base', (0, ix[0], ix[1]))

    @property
    def psi_prim(self, ix=(1, 0)):
        return self.fun_ix('base', (1, ix[0], ix[1]))

    @property
    def phi_dual(self, ix=(1, 0)):
        return self.fun_ix('dual', (0, ix[0], ix[1]))

    @property
    def psi_dual(self, ix=(1, 0)):
        return self.fun_ix('dual', (1, ix[0], ix[1]))

    def fun_ix(self, what, ix=None):
        """
        Returns wave function for given index.
        :param what: Either 'base' or 'dual'; specify the system
        :param ix: the index within the system. Defined as a triple (q, s, z),
            q : either 0=scaling, 1=mother wavelet
            s : scale, usually a power of two (2^j)
            z : offset for given scale
            It is optional, and returns the standard scaling wave at q=1, z=0 for the system
        :return: function object (callable), which can operate over numpy arrays; the
            function object will have an attribute .support with the support at given
            scale 's' and translation 'z'

        Note: s (scale) can't go beyond 64 (j=6), as numerical accuracy is lost
        """
        if ix is None:
            ix = (0, 1, 0)
        q, s, z = ix
        fun = self.funs[what][q]
        a, b = fun.support
        s2 = math.sqrt(s)
        f = lambda x: s2 * fun(s * x + z)
        f.support = ((a - z)/s, (b - z)/s)
        f._ix = ix
        return f

    def supp_ix(self, what, ix=None):
        """
        Returns an indicator function for `fun_ix` with same parameters that operate over numpy arrays
        :param what: Either 'base' or 'dual'
        :param ix: See `fun_ix`
        :return: function object (callable) that is 1 is inside support, 0 otherwise
        """
        if ix is None:
            ix = (0, 1, 0)
        q, s, z = ix
        fun = self.funs[what][q]
        a, b = fun.support
        f = lambda x: (lambda v: np.less(a, v) & np.less(v, b))(s * x + z)
        f._ix = ix
        return f

    def zrange(self, what, xrange, ix=None):
        """
        Returns the range of z values that cover an interval (minx, maxx) for given index ix in system what
        :param what: Either 'base' or 'dual'
        :param xrange: tuple with (min, max) values for x
        :param ix: See `fun_ix`
        :return: tuple
        """
        if ix is None:
            ix = (0, 1, 0)
        q, s, z = ix
        fun = self.funs[what][q]
        a, b = fun.support
        minx, maxx = xrange[0], xrange[1]
        # minx <= x <= maxx
        # a <= s * x + z <= b
        # a - s * x <= z <= b - s * x
        # a - s * maxx <= z <= b - s * maxx
        # a - s * minx <= z <= b - s * minx
        # Hence, a - s * maxx <= z <= b - s * minx
        zmin = math.floor(a - s * maxx)
        zmax = math.ceil(b - s * minx)
        return (zmin, zmax)


class WaveletTensorProduct(object):
    """
    Tensor product wavelet in $R^d$. It supports similar or different wavelets in each dimension.

        wave1 = WaveletTensorProduct(('db4',) * 3) # db4 in all 3 axes
        wave2 = WaveletTensorProduct(('rbio2.4', 'rbio1.3', 'rbio3.5')) # three different spline wavelets
    """
    def __init__(self, wave_names):
        self.dim = len(wave_names)
        self.waves = [Wavelet(name) for name in wave_names]
        self.orthogonal = all([wave.orthogonal for wave in self.waves])
        self.qq = list(itt.product(range(2), repeat=self.dim))
        self.name = 'x'.join(wave_names)

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
        "See Wavelet.fun_ix"
        qq, ss, zz = ix
        supp_min = np.array([self.waves[i].support[what][qq[i]][0] for i in range(self.dim)])
        supp_max = np.array([self.waves[i].support[what][qq[i]][1] for i in range(self.dim)])
        def f(xx):
            proj = self.proj_fun(self.dim, xx)
            resp = None
            for i in range(self.dim):
                col_i = self.waves[i].fun_ix(what, (qq[i], ss[i], zz[i]))(proj(i))
                if resp is None:
                    resp = col_i
                else:
                    resp = np.multiply(resp, col_i)
            return resp
        f.dim = self.dim
        f.support = np.array([supp_min - zz, supp_max - zz]) / ss
        f._ix = ix
        return f

    def supp_ix(self, what, ix):
        qq, ss, zz = ix
        def f(xx):
            proj = self.proj_fun(self.dim, xx)
            resp = None
            for i in range(self.dim):
                col_i = self.waves[i].supp_ix(what, (qq[i], ss[i], zz[i]))(proj(i))
                if resp is None:
                    resp = col_i
                else:
                    resp = resp & col_i
            return resp.astype(int)
        f._ix = ix
        return f

    @staticmethod
    def proj_fun(dim, xx):
        if type(xx) == tuple or type(xx) == list:
            assert len(xx) == dim
            return lambda i: xx[i]
        else:
            return lambda i: xx[:, i]

    def zs_range(self, what, minx, maxx, qs, js):
        zs_min = np.floor(np.array([self.waves[i].support[what][qs[i]][0] - (2 ** js[i]) * maxx[i] for i in range(self.dim)])) - 1
        zs_max = np.ceil(np.array([self.waves[i].support[what][qs[i]][1] - (2 ** js[i]) * minx[i] for i in range(self.dim)])) + 1
        return zs_min, zs_max
