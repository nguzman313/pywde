import math
import numpy as np
import itertools as itt
from .common import all_zs_tensor
from sklearn.neighbors import BallTree
from scipy.special import gamma, loggamma
from datetime import datetime

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
        self.test = None
        self.ball_tree = None
        self.xs_balls = None
        self.xs_balls_inx = None

    def calc_coeffs(self, xs, cv=False):
        self.n = xs.shape[0]
        self.ball_tree = BallTree(xs)
        self.calculate_nearest_balls(xs, cv)
        for key in self.coeffs.keys():
            j, qx, zs, jpow2 = key
            jpow2 = np.array(jpow2)
            num = self.wave.supp_ix('dual', (qx, jpow2, zs))(xs).sum()
            coeff = (self.wave.fun_ix('dual', (qx, jpow2, zs))(xs) * self.xs_balls).sum() * self.omega(self.n)
            #print('beta_{%s}' % str(key), '=', coeff)
            self.coeffs[key] = (coeff, num)
        print('calc_coeffs #', len(self.coeffs))

    def calc_pdf(self, coeffs):
        def fun(coords):
            xs_sum = self.xs_sum_zeros(coords)
            for key in coeffs.keys():
                j, qx, zs, jpow2 = key
                jpow2 = np.array(jpow2)
                coeff, num = coeffs[key]
                vals = coeff * self.wave.fun_ix('base', (qx, jpow2, zs))(coords)
                xs_sum += vals
            return (xs_sum * xs_sum) / fun.norm_const

        if self.wave.orthogonal:
            fun.norm_const = sum([coeff*coeff for coeff, num in coeffs.values()])
        else:
            print('Biorthogonal - need numerical integration')
            fun.norm_const = 1.0
            x = np.linspace(0, 1, 100)
            y = np.linspace(0, 1, 100)
            grid = np.meshgrid(x, y)
            fun.norm_const = fun(grid).mean()
        fun.dim = self.wave.dim
        min_num = min([num for coeff, num in coeffs.values() if num > 0])
        print('>> WDE PDF')
        print('Num coeffs', len(coeffs))
        print('Norm', fun.norm_const)
        print('min num', min_num)
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

    def calc_terms(self, key, coeff, xs):
        # see paper, Q definition
        if self.xs_balls_inx is None:
            raise ValueError('Use calc_coeffs first')

        j, qx, zs, jpow2 = key
        jpow2 = np.array(jpow2)

        fun_i_dual = self.wave.fun_ix('dual', (qx, jpow2, zs))(xs)
        fun_i_base = self.wave.fun_ix('base', (qx, jpow2, zs))(xs)
        omega_n = self.omega(self.n)
        omega_n2 = omega_n * omega_n

        term1 = coeff * coeff

        term2 = omega_n2 * (fun_i_dual * fun_i_base * self.xs_balls * self.xs_balls).sum()

        vals_i = np.zeros(self.n)
        for j in range(self.n):
            i = self.xs_balls_inx[j,-1]
            psi_i = fun_i_base[i]
            psi_j = fun_i_dual[j]
            v1_i = self.xs_balls[i]
            deltaV_j = self.xs_balls2[j] - self.xs_balls[i]
            v = psi_i * psi_j * v1_i * deltaV_j
            vals_i[i] += v
        term3 = omega_n2 * vals_i.sum()
        return term1, term2, term3

    def calc_contribution(self, key, coeff, xs):
        omega_n1 = self.omega(self.n - 1)
        omega_n = self.omega(self.n)
        term1, term2, term3 = self.calc_terms(key, coeff, xs)
        return omega_n1 * (term1 - term2 + term3) / omega_n

    def calc_b2_correction(self, key, coeff, xs):
        term1, term2, term3 = self.calc_terms(key, coeff, xs)
        return term2, term3

    def calc1(self, key, tup, coords, norm):
        j, qx, zs, jpow2 = key
        jpow2 = np.array(jpow2)
        coeff, num = tup
        vals = coeff * self.wave.fun_ix('base', (qx, jpow2, zs))(coords)
        return vals, norm + coeff*coeff

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

    def sqrt_vunit(self):
        "Volume of unit hypersphere in d dimensions"
        return (np.pi ** (self.wave.dim / 2.0)) / gamma(self.wave.dim / 2.0 + 1)

    def omega(self, n):
        "Bias correction for k-th nearest neighbours sum for sample size n"
        return gamma(self.k) / gamma(self.k + 0.5) / math.sqrt(n)

    def calculate_nearest_balls(self, xs, cv):
        if cv:
            k = self.k + 1
            ix = -2
        else:
            k = self.k
            ix = -1
        dist, inx = self.ball_tree.query(xs, k + 1)
        k_near_radious = dist[:, ix:]
        xs_balls = np.power(k_near_radious, self.wave.dim / 2.0)
        self.xs_balls = xs_balls[:, ix] * self.sqrt_vunit()
        if cv:
            self.xs_balls2 = xs_balls[:, -1] * self.sqrt_vunit()
            self.xs_balls_inx = inx

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

    def _fitinit(self, xs, cv=False):
        if self.wave.dim != xs.shape[1]:
            raise ValueError("Expected data with %d dimensions, got %d" % (self.wave.dim, xs.shape[1]))
        self.minx = np.amin(xs, axis=0)
        self.maxx = np.amax(xs, axis=0)
        self.params = WParams(self)
        self.params.calc_coeffs(xs, cv)

    def fit(self, xs):
        "Fit estimator to data. xs is a numpy array of dimension n x d, n = samples, d = dimensions"
        print('Regular estimator')
        t0 = datetime.now()
        self._fitinit(xs)
        self.pdf = self.params.calc_pdf(self.params.coeffs)
        self.name = '%s, n=%d, j0=%s, Dj=%d FIT' % (self.wave.name, self.params.n, str(self.jj0), self.delta_j)
        print('secs=', (datetime.now() - t0).total_seconds())

    def cvfit(self, xs):
        print('CV estimator')
        t0 = datetime.now()
        self._fitinit(xs, cv=True)
        coeffs = self.calc_pdf_cv(xs)
        self.pdf = self.params.calc_pdf(coeffs)
        self.name = '%s, n=%d, j0=%s, Dj=%d CV #params=%d' % (self.wave.name, self.params.n, str(self.jj0),
                                                              self.delta_j, len(coeffs))
        print('secs=', (datetime.now() - t0).total_seconds())

    def calc_pdf_cv(self, xs):
        coeffs = {}
        for key_tup in self.params.coeffs.items():
            key, tup = key_tup
            coeff, num = tup
            if coeff == 0.0:
                continue
            coeff_contribution = self.params.calc_contribution(key, coeff, xs)
            if coeff_contribution > 0:
                coeffs[key] = tup
        return coeffs

    def mdlfit(self, xs):
        print('MDL-like estimator')
        t0 = datetime.now()
        self._fitinit(xs, cv=True)
        coeffs = self.calc_pdf_mdl(xs)
        self.pdf = self.params.calc_pdf(coeffs)
        self.name = '%s, n=%d, j0=%s, Dj=%d CV-like #params=%d' % (self.wave.name, self.params.n, str(self.jj0),
                                                              self.delta_j, len(coeffs))
        print('secs=', (datetime.now() - t0).total_seconds())

    def calc_pdf_mdl(self, xs):
        all_coeffs = []
        for key_tup in self.params.coeffs.items():
            key, tup = key_tup
            coeff, _ = tup
            if coeff == 0.0:
                continue
            coeff_contribution = self.params.calc_contribution(key, coeff, xs)
            all_coeffs.append((key, coeff_contribution))

        # sort 1 : lambda (key, Q): -Q
        all_coeffs.sort(key=lambda t: -t[1])

        # sort 2 : lambda (key, Q): (is_beta, j, -Q)
        # is_alpha = lambda key: (lambda j, qx: j == 0 and all([qi == 0 for qi in qx]))(key[0], key[1])
        # all_coeffs.sort(key=lambda t: (not is_alpha(t[0]), t[0][0], -t[1]))
        keys = []
        for key, contrib in all_coeffs: ##[:self.params.n]:
            keys.append(key)
            # if is_alpha(key): # sort 2
            #     continue
            if math.fabs(contrib) < 0.00001:
                break
        return {key:self.params.coeffs[key] for key in keys}


def coeff_sort(key_tup):
    key, tup = key_tup
    j, qx, zs, jpow2 = key
    coeff, num = tup
    is_alpha = j == 0 and all([qi == 0 for qi in qx])
    v_th = math.fabs(coeff) / math.sqrt(j + 1)
    return (not is_alpha, -v_th, key)

def coeff_sort_no_j(key_tup):
    key, tup = key_tup
    j, qx, zs, jpow2 = key
    coeff, num = tup
    is_alpha = j == 0 and all([qi == 0 for qi in qx])
    v_th = math.fabs(coeff)
    return (not is_alpha, -v_th, key)

def _cv2_key_sort(key):
    j, qx, zs, jpow2 = key
    is_alpha = j == 0 and all([qi == 0 for qi in qx])
    return (not is_alpha, -j, qx, zs)
