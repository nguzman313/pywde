import math
import numpy as np
import itertools as itt
from scipy.interpolate import interp1d
from functools import partial
from .common import all_zs_tensor
from sklearn.neighbors import BallTree
from scipy.special import gamma

from .pywt_ext import WaveletTensorProduct


class WaveletDensityEstimator(object):
    def __init__(self, waves, k=1, delta_j=0, thresholding=None):
        """
        Builds a shape-preserving estimator based on square root and nearest neighbour distance.

        :param waves: wave specification for each dimension: List of (wave_name:str, j0:int)
        :param k: use k-th neighbour
        :param: delta_j: number of levels to go after j0 on the wavelet expansion part; 0 means no wavelet expansion,
            only scaling functions.
        """
        self.wave = WaveletTensorProduct([wave_desc[0] for wave_desc in waves])
        self.k = k
        self.jj0 = np.array([wave_desc[1] for wave_desc in waves])
        self.delta_j = delta_j + 1
        self.pdf = None
        # TODO: move into `waves`
        if thresholding is None:
            self.thresholding = lambda n, j, dn, c: c
        else:
            self.thresholding = thresholding

    def fit(self, xs):
        "Fit estimator to data. xs is a numpy array of dimension n x d, n = samples, d = dimensions"
        if self.wave.dim != xs.shape[1]:
            raise ValueError("Expected data with %d dimensions, got %d" % (self.wave.dim, xs.shape[1]))
        self.minx = np.amin(xs, axis=0)
        self.maxx = np.amax(xs, axis=0)
        self.n = xs.shape[0]
        self.calc_coefficients(xs)
        self.pdf = self.calc_pdf()
        return True

    def calc_coefficients(self, xs):
        xs_balls = self.calculate_nearest_balls(xs)
        self.do_calculate(xs, xs_balls)

    def do_calculate(self, xs, xs_balls):
        self.coeffs = {}
        self.nums = {}
        qq = self.wave.qq
        self.do_calculate_j(0, qq[0:1], xs, xs_balls)
        for j in range(self.delta_j):
            self.do_calculate_j(j, qq[1:], xs, xs_balls)

    def do_calculate_j(self, j, qxs, xs, xs_balls):
        if j not in self.coeffs:
            self.coeffs[j] = {}
            self.nums[j] = {}
        jj = self._jj(j)
        jpow2 = 2 ** jj
        for qx in qxs:
            zs_min, zs_max = self.wave.zs_range('dual', self.minx, self.maxx, qx, jj)
            self.coeffs[j][qx] = {}
            self.nums[j][qx] = {}
            for zs in itt.product(*all_zs_tensor(zs_min, zs_max)):
                num = self.wave.supp_ix('dual', (qx, jpow2, zs))(xs).sum()
                self.nums[j][qx][zs] = num
                v = (self.wave.fun_ix('dual', (qx, jpow2, zs))(xs) * xs_balls[:,0]).sum()
                self.coeffs[j][qx][zs] = v

    def calc_pdf(self):
        def pdffun_j(coords, xs_sum, j, qxs, threshold):
            jj = self._jj(j)
            jpow2 = 2 ** jj
            norm_j = 0.0
            for qx in qxs:
                for zs, coeff in self.coeffs[j][qx].items():
                    # num = self.nums[j][qx][zs]
                    # TODO: self.thresholding(self.n, j, num, coeff) if threshold and self.thresholding else coeff
                    coeff_t = coeff
                    norm_j += coeff_t * coeff_t
                    vals = coeff_t * self.wave.fun_ix('base', (qx, jpow2, zs))(coords)
                    xs_sum += vals
            return norm_j
        def pdffun(coords):
            if type(coords) == tuple or type(coords) == list:
                xs_sum = np.zeros(coords[0].shape, dtype=np.float64)
            else:
                xs_sum = np.zeros(coords.shape[0], dtype=np.float64)
            qq = self.wave.qq
            norm_const = pdffun_j(coords, xs_sum, 0, qq[0:1], False)
            for j in range(self.delta_j):
                norm_const += pdffun_j(coords, xs_sum, j, qq[1:], True)
            return (xs_sum * xs_sum)/norm_const
        pdffun.dim = self.wave.dim
        return pdffun

    def _jj(self, j):
        return np.array([j0 + j for j0 in self.jj0])

    # factor for num samples n, dimension dim and nearest index k
    def calc_factor(self):
        v_unit = (np.pi ** (self.wave.dim / 2.0)) / gamma(self.wave.dim / 2.0 + 1)
        return math.sqrt(v_unit) * (gamma(self.k) / gamma(self.k + 0.5)) / math.sqrt(self.n)

    def calculate_nearest_balls(self, xs):
        ball_tree = BallTree(xs)
        k_near_radious = ball_tree.query(xs, self.k + 1)[0][:, [-1]]
        factor = self.calc_factor()
        return np.power(k_near_radious, self.wave.dim / 2.0) * factor



