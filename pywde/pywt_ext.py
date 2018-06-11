"""
Extensions to PyWavelets (pywt) to calculate wavelet values
"""
import math
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


class Wavelet(pywt.Wavelet):
    """Wrapper around pywt.Wavelet that defines support, phi, psi methods for the base wavelets and
    corresponding duals. If they are orthogonal base and duals are the same. The methods
    work on 1D numpy arrays iterating over elements. For consistency in terminology, use duals to
    calculate coefficients and base to reconstruct the signal (e.g. eq (3.22) in
    1992, Chui, Wang, "On Compactly Supported Spline Wavelets and a Duality Principle")
    """
    def __init__(self, wave_name):
        self.wave = pywt.Wavelet(wave_name)
        self.support = self.wave_support_info(self.wave)
        self.funs = {}
        self.calc_wavefuns()

    @staticmethod
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
        elif pywt_wave.family_name == 'Biorthogonal':
            # pywt uses Spline Wavelets
            # base is the reconstruction family, dual the decomposition family
            dec_vm = int(pywt_wave.name[6:])  # decomposition order
            rec_vm = int(pywt_wave.name[4:5]) # reconstruction order
            # support for lowpass in primal and dual as calculated by Mathematica
            n1, n2 = (-dec_vm - (rec_vm - 1) //2 + (dec_vm % 2), dec_vm + (rec_vm - 1) // 2)
            nd1, nd2 = (-(rec_vm  + 1)// 2, (rec_vm  + 1)// 2)
            # from this, we can infer support for highpass for base and dual
            resp['base'] = ((n1, n2), ((n1 - nd2 + 1)//2, (n2 - nd1 + 1)//2))
            resp['dual'] = ((nd1, nd2), ((nd1 - n2 + 1)//2, (nd2 - n1 + 1)//2))
        else:
            raise ValueError('pywt_wave family %s not known support' % pywt_wave.family_name)
        return resp

    @staticmethod
    def trim_zeros(coeffs):
        nz = np.nonzero(coeffs)
        return coeffs[np.min(nz):np.max(nz)+1]

    def calc_wavefuns(self):
        values = self.wave.wavefun(level=14)
        phi_support_d, psi_support_d = self.support['base']
        phi_support_r, psi_support_r = self.support['dual']
        if len(values) == 5:
            phi_d, psi_d, phi_r, psi_r, _ = values
            # biorthogonal wavelets have zeros in pywt for some reason; we have to remove them
            # to match the support
            phi_d, psi_d, phi_r, psi_r = [self.trim_zeros(c) for c in [phi_d, psi_d, phi_r, psi_r]]
        else:
            phi_d, psi_d, _ = values
            phi_r, psi_r = phi_d, psi_d
        # reconstruction '_r' is the base
        phi = self.calc_fun(phi_support_r, phi_r)
        psi = self.calc_fun(psi_support_r, psi_r)
        self.funs['base'] = (phi, psi)
        # decomposition '_d' is the dual
        phi = self.calc_fun(phi_support_d, phi_d)
        psi = self.calc_fun(psi_support_d, psi_d)
        self.funs['dual'] = (phi, psi)

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

    @staticmethod
    def calc_fun(support, values):
        resp = interp1d(np.linspace(*support, num=len(values)), values, fill_value=0.0, bounds_error=False, kind=1)
        resp.support = support
        return resp

class WaveletTensorProduct(object):
    def __init__(self, wave_names):
        self.dim = len(wave_names)
        self.waves = [Wavelet(name) for name in wave_names]
        self.qq = list(itt.product(range(2), repeat=self.dim))

    def prim(self, ix):
        return self.fun_ix('base', ix)

    def dual(self, ix):
        return self.fun_ix('dual', ix)

    def fun_ix(self, what, ix):
        qq, ss, zz = ix
        ss2 = math.sqrt(np.prod(ss))
        def f(xx):
            cols = []
            for i, q2 in enumerate(qq):
                wave = self.waves[i].funs[what]
                xs_proj = xx[:,i] # proj(xs, i)
                cols.append(wave[q2](ss[i] * xs_proj - zz[i]))
            return np_mult(tuple(cols)) * ss2
        return f
