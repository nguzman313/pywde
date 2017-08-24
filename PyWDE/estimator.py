from __future__ import division
import pywt
import numpy as np
import itertools as itt
from scipy.interpolate import interp1d
from functools import partial
from .common import *

class WaveletDensityEstimator(object):
    def __init__(self, wave_name, k=1, j0=1, j1=None, thresholding=None):
        self.wave = pywt.Wavelet(wave_name)
        self.k = k
        self.j0 = j0
        self.j1 = j1 if j1 is not None else (j0 - 1)
        self.multi_supports = wave_support_info(self.wave)
        self.pdf = None
        if thresholding is None:
            self.thresholding = lambda n, j, dn, c: c
        else:
            self.thresholding = thresholding

    def fit(self, xs):
        "Fit estimator to data. xs is a numpy array of dimension n x d, n = samples, d = dimensions"
        self.dim = xs.shape[1]
        self.dimpow = 2 ** self.dim
        self.set_wavefuns(self.dim)
        self.minx = np.amin(xs, axis=0)
        self.maxx = np.amax(xs, axis=0)
        self.n = xs.shape[0]
        self.calc_coefficients(xs)
        self.pdf = self.calc_pdf()
        return True

    def set_wavefuns(self, dim):
        self.wave_funs = self.calc_wavefuns(dim, self.multi_supports['base'], self.wave)
        self.dual_wave_funs = self.calc_wavefuns(dim, self.multi_supports['dual'], self.wave)

    @staticmethod
    def calc_wavefuns(dim, supports, wave):
        resp = {}
        phi_support, psi_support = supports
        phi, psi, _ = wave.wavefun(level=12)
        phi = interp1d(np.linspace(*phi_support, num=len(phi)), phi, fill_value=0.0, bounds_error=False)
        psi = interp1d(np.linspace(*psi_support, num=len(psi)), psi, fill_value=0.0, bounds_error=False)
        for wave_x, qx in all_qx(dim):
            f = partial(wave_tensor, qx, phi, psi)
            f.qx = qx
            f.support = support_tensor(qx, phi_support, psi_support)
            f.suppf = partial(suppf_tensor, qx, phi_support, psi_support)
            resp[tuple(qx)] = f
        return resp

    def calc_coefficients(self, xs):
        #grid_xs = gridify_xs(self.j0, self.j1, xs, self.minx, self.maxx)
        xs_balls = calculate_nearest_balls(self.k, xs)
        self.do_calculate(xs, xs_balls)

    def do_calculate(self, xs, xs_balls):
        self.coeffs = {}
        self.nums = {}
        qxs = list(all_qx(self.dim))
        self.do_calculate_j(self.j0, qxs[0:1], xs, xs_balls)
        for j in range(self.j0, self.j1 + 1):
            self.do_calculate_j(j, qxs[1:], xs, xs_balls)

    def do_calculate_j(self, j, qxs, xs, xs_balls):
        jpow2 = 2 ** j
        if j not in self.coeffs:
            self.coeffs[j] = {}
            self.nums[j] = {}
        for ix, qx in qxs:
            wavef = self.wave_funs[qx]
            zs_min, zs_max = zs_range(wavef, self.minx, self.maxx, j)
            self.coeffs[j][qx] = {}
            self.nums[j][qx] = {}
            for zs in itt.product(*all_zs_tensor(zs_min, zs_max)):
                num = calc_num(wavef.suppf, jpow2, zs, xs)
                self.nums[j][qx][zs] = num
                v = calc_coeff(wavef, jpow2, zs, xs, xs_balls)
                self.coeffs[j][qx][zs] = v

    def get_betas(self, j):
        return [coeff for ix, qx in list(all_qx(self.dim))[1:] for coeff in self.coeffs[j][qx].values()]

    def get_nums(self):
        return [coeff
                for j in self.nums
                    for ix, qx in list(all_qx(self.dim))[1:]
                        for coeff in self.nums[j][qx].values()]

    def calc_pdf(self):
        def pdffun_j(coords, xs_sum, j, qxs, threshold):
            jpow2 = 2 ** j
            norm_j = 0.0
            for ix, qx in qxs:
                wavef = self.dual_wave_funs[qx]
                for zs, coeff in self.coeffs[j][qx].iteritems():
                    num = self.nums[j][qx][zs]
                    coeff_t = self.thresholding(self.n, j - self.j0, num, coeff) if threshold else coeff
                    norm_j += coeff_t * coeff_t
                    vals = coeff_t * wavef(jpow2, zs, coords)
                    xs_sum += vals
            return norm_j
        def pdffun(coords):
            xs_sum = np.zeros(coords[0].shape, dtype=np.float64)
            qxs = list(all_qx(self.dim))
            norm_const = pdffun_j(coords, xs_sum, self.j0, qxs[0:1], False)
            for j in range(self.j0, self.j1 + 1):
                norm_const += pdffun_j(coords, xs_sum, j, qxs[1:], True)
            return (xs_sum * xs_sum)/norm_const
        return pdffun
