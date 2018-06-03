"""
Extensions to PyWavelets (pywt) to calculate wavelet values
"""
import pywt
from scipy.interpolate import interp1d
import numpy as np

def affine1d(poj, z, x):
    return poj * x - z


class Wavelet(pywt.Wavelet):
    """Wrapper around pywt.Wavelet that defines support, phi, psi methods for wavelet and
    corresponding duals if they are biorthogonal instead of orthogonal. The methods
    work as
    """
    def __init__(self, wave_name):
        self.wave = pywt.Wavelet(wave_name)
        self.support = self.wave_support_info(self.wave)
        self.phi, self.psi = self.calc_wavefuns(self.support['base'], self.wave)
        self.phi_dual, self.psi_dual = self.calc_wavefuns(self.support['dual'], self.wave)

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
            phi_support = (1 - pywt_wave.dec_len // 2, pywt_wave.dec_len // 2)
            psi_support = (1 - pywt_wave.dec_len // 2, pywt_wave.dec_len // 2)
            resp['base'] = (phi_support, psi_support)
            raise ValueError('pywt_wave family %s not known support' % pywt_wave.family_name)
        else:
            raise ValueError('pywt_wave family %s not known support' % pywt_wave.family_name)
        return resp

    @staticmethod
    def calc_wavefuns(supports, wave):
        phi_support, psi_support = supports
        phi, psi, _ = wave.wavefun(level=10)
        print('len(phi)', len(phi))
        kind = 1 if wave.dec_len <= 4 else 2
        phi_fun = interp1d(np.linspace(*phi_support, num=len(phi)), phi, fill_value=0.0, bounds_error=False, kind=kind)
        phi_fun.support = phi_support
        psi_fun = interp1d(np.linspace(*psi_support, num=len(psi)), psi, fill_value=0.0, bounds_error=False, kind=kind)
        psi_fun.support = psi_support
        return (phi_fun, psi_fun)
