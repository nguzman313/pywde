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


class WParams(object):
    def __init__(self, wde):
        self.k = wde.k
        self.wave = wde.wave
        self.jj0 = wde.jj0
        self.delta_j = wde.delta_j
        self.coeffs = {}
        self.minx = wde.minx
        self.maxx = wde.maxx
        self._calc_indexes()
        self.n = 0
        self.pdf = None

    def calc_coeffs(self, xs):
        self.n = xs.shape[0]
        xs_balls = self._calculate_nearest_balls(xs)
        for key in self.coeffs.keys():
            j, qx, zs, jpow2 = key
            jpow2 = np.array(jpow2)
            num = self.wave.supp_ix('dual', (qx, jpow2, zs))(xs).sum()
            coeff = (self.wave.fun_ix('dual', (qx, jpow2, zs))(xs) * xs_balls[:, 0]).sum()
            self.coeffs[key] = (coeff, num)

    def calc_pdf(self, coeffs):
        norm_const = sum([coeff*coeff for coeff, num in coeffs.values()])

        def fun(coords):
            xs_sum = self.xs_sum_zeros(coords)
            for key in coeffs.keys():
                j, qx, zs, jpow2 = key
                jpow2 = np.array(jpow2)
                coeff, num = coeffs[key]
                vals = coeff * self.wave.fun_ix('base', (qx, jpow2, zs))(coords)
                xs_sum += vals
            return (xs_sum * xs_sum) / norm_const

        fun.dim = self.wave.dim
        return fun

    def gen_pdf(self, xs_sum, coeffs_items, coords):
        norm_const = 0.0
        for key, tup in coeffs_items:
            j, qx, zs, jpow2 = key
            jpow2 = np.array(jpow2)
            coeff, num = tup
            vals = coeff * self.wave.fun_ix('base', (qx, jpow2, zs))(coords)
            norm_const += coeff * coeff
            xs_sum += vals
            yield key, (xs_sum * xs_sum) / norm_const

    def xs_sum_zeros(self, coords):
        if type(coords) == tuple or type(coords) == list:
            xs_sum = np.zeros(coords[0].shape, dtype=np.float64)
        else:
            xs_sum = np.zeros(coords.shape[0], dtype=np.float64)
        return xs_sum

    def _calc_indexes(self):
        qq = self.wave.qq
        self._calc_indexes_j(0, qq[0:1])
        for j in range(self.delta_j):
            self._calc_indexes_j(j, qq[1:])

    def _calc_indexes_j(self, j, qxs):
        jj = self._jj(j)
        jpow2 = tuple(2 ** jj)
        for qx in qxs:
            zs_min, zs_max = self.wave.zs_range('dual', self.minx, self.maxx, qx, jj)
            for zs in itt.product(*all_zs_tensor(zs_min, zs_max)):
                self.coeffs[(j, qx, zs, jpow2)] = None

    # factor for num samples n, dimension dim and nearest index k
    def _calc_factor(self):
        v_unit = (np.pi ** (self.wave.dim / 2.0)) / gamma(self.wave.dim / 2.0 + 1)
        return math.sqrt(v_unit) * (gamma(self.k) / gamma(self.k + 0.5)) / math.sqrt(self.n)

    def _calculate_nearest_balls(self, xs):
        ball_tree = BallTree(xs)
        k_near_radious = ball_tree.query(xs, self.k + 1)[0][:, [-1]]
        factor = self._calc_factor()
        return np.power(k_near_radious, self.wave.dim / 2.0) * factor

    def _jj(self, j):
        return np.array([j0 + j for j0 in self.jj0])

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
        self.delta_j = delta_j
        self.wave_series = None
        self.pdf = None
        self.thresholding = None
        self.params = None

    def _fitinit(self, xs, cv=None):
        if self.wave.dim != xs.shape[1]:
            raise ValueError("Expected data with %d dimensions, got %d" % (self.wave.dim, xs.shape[1]))
        self.minx = np.amin(xs, axis=0)
        self.maxx = np.amax(xs, axis=0)
        if cv is None:
            self.params = WParams(self)
            self.params.calc_coeffs(xs)
        else:
            self.params = [WParams(self) for k in range(cv)]

    def fit(self, xs):
        "Fit estimator to data. xs is a numpy array of dimension n x d, n = samples, d = dimensions"
        self._fitinit(xs)
        self.pdf = self.params.calc_pdf(self.params.coeffs) ##self.calc_pdf()
        self.name = '%s, n=%d, j0=%s, Dj=%d' % (self.wave.name, self.params.n, str(self.jj0), self.delta_j)
        return True

    def mdlfit(self, xs):
        self._fitinit(xs)
        self.calc_pdf_mdl(xs)
        self.name = '%s, n=%d, j0=%s, Dj=%d' % (self.wave.name, self.params.n, str(self.jj0), self.delta_j)
        return True

    def calc_pdf_mdl(self, xs):
        all_coeffs = list(self.params.coeffs.items())
        def coeff_sort(key_tup):
            key, tup = key_tup
            j, qx, zs, jpow2 = key
            coeff, num = tup
            is_alpha = j == 0 and all([qi == 0 for qi in qx])
            v_th = math.fabs(coeff) / math.sqrt(j + 1)
            return (not is_alpha, -v_th, key)
        all_coeffs.sort(key=coeff_sort)
        xs_sum = self.params.xs_sum_zeros(xs)
        k = 0
        keys = []
        ranking = []
        best_mdl = best_k = None
        for key, pdf_for_xs in self.params.gen_pdf(xs_sum, all_coeffs[:self.params.n], xs):
            j, qx, zs, jpow2 = key
            is_alpha = j == 0 and all([qi == 0 for qi in qx])
            k += 1
            keys.append(key)
            if is_alpha:
                continue
            logLL = - np.log(pdf_for_xs).sum()
            penalty = log_riemann_volume_class(k) - log_riemann_volume_param(k, self.params.n)
            if best_mdl is None or logLL + penalty < best_mdl:
                best_mdl = logLL + penalty
                best_k = len(keys)
            rank_tuple = (k, logLL, penalty, logLL + penalty)
            ranking.append(rank_tuple)
        print('MDL all_params=',len(all_coeffs), ' best k=', best_k, 'best MDL=', best_mdl)
        coeffs = {key:self.params.coeffs[key] for key in keys[:best_k]}
        self.pdf = self.params.calc_pdf(coeffs)
        self.ranking = ranking
