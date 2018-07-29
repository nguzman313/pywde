import math
import numpy as np
import itertools as itt
from .common import all_zs_tensor
from sklearn.neighbors import BallTree
from scipy.special import gamma, loggamma

from .pywt_ext import WaveletTensorProduct


def log_factorial(n):
    if n <= 1:
        return 0
    return np.log(np.array(range(2, n + 1))).sum()

def log_riemann_volume_class(k):
    "Total Riemannian volume in model class with k parameters"
    return math.log(2) + k/2 * math.log(math.pi) - loggamma(k/2)

def log_riemann_volume_param(k, n):
    "Riemannian volume around estimate with k parameters for n samples"
    return (k/2) * math.log(2 * math.pi / n)

class WaveletDensityEstimator(object):
    def __init__(self, waves, k=1, delta_j=0):
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
        self.thresholding = None

    def fit(self, xs):
        "Fit estimator to data. xs is a numpy array of dimension n x d, n = samples, d = dimensions"
        if self.wave.dim != xs.shape[1]:
            raise ValueError("Expected data with %d dimensions, got %d" % (self.wave.dim, xs.shape[1]))
        self.minx = np.amin(xs, axis=0)
        self.maxx = np.amax(xs, axis=0)
        self.n = xs.shape[0]
        self.calc_coefficients(xs)
        self.pdf = self.calc_pdf()
        self.name = '%s, n=%d, j0=%s, Dj=%d' % (self.wave.name, self.n, str(self.jj0), self.delta_j)
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
            if self.thresholding and threshold:
                th_fun = self.thresholding
            else:
                th_fun = lambda n, j, dn, c: c
            for qx in qxs:
                for zs, coeff in self.coeffs[j][qx].items():
                    num = self.nums[j][qx][zs]
                    coeff_t = th_fun(self.n, j, num, coeff)
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

    def mdl(self, xs):
        alphas, betas = self.k_range()
        best = (float('inf'), None, None)
        num_betas = len(betas)
        self.mld_data = {'k_betas': [], 'logLL': [], 'MLD_penalty': []}
        for betas_num, th_value in enumerate(betas):
            if th_value == 0.0:
                continue
            k = alphas + (num_betas - betas_num)
            penalty = log_riemann_volume_class(k) - log_riemann_volume_param(k, self.n)
            if penalty < 0:
                print('neg')
                continue
            self.thresholding = self.calc_hard_threshold_fun(betas_num, th_value)
            self.pdf = self.calc_pdf()
            logLL = - np.log(self.pdf(xs)).sum()
            val = logLL + penalty
            print(num_betas - betas_num, th_value, val, penalty)
            self.mld_data['k_betas'].append(num_betas - betas_num)
            self.mld_data['logLL'].append(logLL)
            self.mld_data['MLD_penalty'].append(penalty)
            if val < best[0]:
                best = (val, self.thresholding, self.pdf)
                print('>>>>', self.thresholding.__doc__, val)
        self.mld_best = best
        _, self.thresholding, self.pdf = best

    def k_range(self):
        "returns range of valid k (parameters) value"
        # it cannot be greater than number of samples
        # it cannot be greater than the number of coefficients
        qq = self.wave.qq
        alphas = len(self.coeffs[0][qq[0]])
        coeffs = []
        for j in range(self.delta_j):
            vs = itt.chain.from_iterable([self.coeffs[j][qx].values() for qx in qq[1:]])
            coeffs += [(math.fabs(value) / math.sqrt(j + 1)) for value in vs]
        return alphas, sorted(coeffs)

    def calc_hard_threshold_fun(self, betas_num, th_value):
        def soft_th(n, j, dn, coeff):
            lvl_t = th_value * math.sqrt(j + 1)
            if coeff < 0:
                if -coeff < lvl_t:
                    return 0
                else:
                    return coeff + lvl_t
            else:
                if coeff < lvl_t:
                    return 0
                else:
                    return coeff - lvl_t
        soft_th.__doc__ = "Soft threshold at %g (index %d)" % (th_value, betas_num)
        return soft_th
