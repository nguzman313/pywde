import math
import numpy as np
import itertools as itt
from collections import namedtuple
from .common import all_zs_tensor
from sklearn.neighbors import BallTree
from scipy.special import gamma
from datetime import datetime

from .pywt_ext import WaveletTensorProduct

ThresholdResult = namedtuple('ThresholdResult', ['threshold', 'pos_k', 'values', 'msg'])

# from scipy cookbook
def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = getattr(np, window)(window_len)

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y

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

    def calc_coeffs(self, xs):
        self.n = xs.shape[0]
        self.ball_tree = BallTree(xs)
        self.calculate_nearest_balls(xs, True)
        norm = 0.0
        omega = self.omega(self.n)
        for key in self.coeffs.keys():
            j, qx, zs, jpow2 = key
            jpow2 = np.array(jpow2)
            num = self.wave.supp_ix('dual', (qx, jpow2, zs))(xs).sum()
            terms_d = self.wave.fun_ix('dual', (qx, jpow2, zs))(xs)
            terms_b = self.wave.fun_ix('base', (qx, jpow2, zs))(xs)
            coeff = (terms_d * self.xs_balls).sum() * omega
            coeff_b = (terms_b * self.xs_balls).sum() * omega
            #print('beta_{%s}' % str(key), '=', coeff)
            self.coeffs[key] = (coeff, coeff_b, num)
            norm += coeff * coeff_b
        print('calc_coeffs #', len(self.coeffs), norm)

    def calc_pdf(self, coeffs):
        def fun(coords):
            xs_sum = self.xs_sum_zeros(coords)
            for key in coeffs.keys():
                j, qx, zs, jpow2 = key
                jpow2 = np.array(jpow2)
                coeff, coeff_b, num = coeffs[key]
                vals = coeff * self.wave.fun_ix('base', (qx, jpow2, zs))(coords)
                xs_sum += vals
            return (xs_sum * xs_sum) / fun.norm_const

        fun.norm_const = sum([coeff * coeff_b for coeff, coeff_b, num in coeffs.values()])
        fun.dim = self.wave.dim
        fun.nparams = len(coeffs)
        min_num = min([num for coeff, coeff_b, num in coeffs.values() if num > 0])
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
            coeff, coeff_b, num = tup
            vals = coeff * self.wave.fun_ix('base', (qx, jpow2, zs))(coords)
            norm_const += coeff * coeff_b
            xs_sum += vals
            yield key, (xs_sum * xs_sum) / norm_const

    def calc_terms(self, key, coeff, coeff_b, xs):
        # see paper, Q definition
        if self.xs_balls_inx is None:
            raise ValueError('Use calc_coeffs first')

        j, qx, zs, jpow2 = key
        jpow2 = np.array(jpow2)

        fun_i_dual = self.wave.fun_ix('dual', (qx, jpow2, zs))(xs)
        fun_i_base = self.wave.fun_ix('base', (qx, jpow2, zs))(xs)
        omega_n = self.omega(self.n)
        omega_n1 = self.omega(self.n-1)
        omega_n2 = omega_n * omega_n1

        term1 = omega_n1 / omega_n * coeff * coeff_b

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
        return term1, term2, term3, coeff * coeff_b

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
        print('\n')
        self._calc_indexes_j(0, qq[0:1])
        print('# alphas =', len(self.coeffs.keys()))
        for j in range(self.delta_j):
            self._calc_indexes_j(j, qq[1:])
            print('# coeffs %d =' % j, len(self.coeffs.keys()))

    def _calc_indexes_j(self, j, qxs):
        jj = self._jj(j)
        jpow2 = tuple(2 ** jj)
        for qx in qxs:
            zs_min_d, zs_max_d = self.wave.z_range('dual', (qx, jpow2, None), self.minx, self.maxx)
            zs_min_b, zs_max_b = self.wave.z_range('base', (qx, jpow2, None), self.minx, self.maxx)
            zs_min = np.min((zs_min_d, zs_min_b), axis=0)
            zs_max = np.max((zs_max_d, zs_max_b), axis=0)
            for zs in itt.product(*all_zs_tensor(zs_min, zs_max)):
                self.coeffs[(j, qx, zs, jpow2)] = None

    def sqrt_vunit(self):
        "Volume of unit hypersphere in d dimensions"
        return (np.pi ** (self.wave.dim / 4.0)) / gamma(self.wave.dim / 2.0 + 1)

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
        self._xs = None

    def _fitinit(self, xs):
        if self._xs is xs:
            # objec ref comparisson, do not recalc if already calculated
            self.params = self._params
            return
        if self.wave.dim != xs.shape[1]:
            raise ValueError("Expected data with %d dimensions, got %d" % (self.wave.dim, xs.shape[1]))
        self.minx = np.amin(xs, axis=0)
        self.maxx = np.amax(xs, axis=0)
        self.params = WParams(self)
        self.params.calc_coeffs(xs)
        self._xs = xs
        self._params = self.params

    def fit(self, xs):
        "Fit estimator to data. xs is a numpy array of dimension n x d, n = samples, d = dimensions"
        print('Regular estimator')
        t0 = datetime.now()
        self._fitinit(xs)
        self.pdf = self.params.calc_pdf(self.params.coeffs)
        self.name = '%s, n=%d, j0=%s, Dj=%d FIT #params=%d' % (self.wave.name, self.params.n, str(self.jj0),
                                                               self.delta_j, len(self.params.coeffs))
        print('secs=', (datetime.now() - t0).total_seconds())

    def cvfit(self, xs, loss, ordering, is_single=True):
        "options = dict(loss=?, ordering=?)"
        if loss not in WaveletDensityEstimator.LOSSES:
            raise ValueError('Wrong loss')
        if ordering not in WaveletDensityEstimator.ORDERINGS:
            raise ValueError('Wrong ordering')
        print('CV estimator: %s, %s; single %s' % (loss, ordering, str(is_single)))
        t0 = datetime.now()
        self._fitinit(xs)
        coeffs = self.calc_pdf_cv(xs, loss, ordering, is_single)
        self.pdf = self.params.calc_pdf(coeffs)
        self.name = '%s, n=%d, j0=%s, Dj=%d #params=%d Lss=%s Ord=%s' % (self.wave.name, self.params.n,
                                                                             str(self.jj0), self.delta_j, len(coeffs),
                                                                             loss[:3], ordering[:3])
        print('secs=', (datetime.now() - t0).total_seconds())

    Q_ORD = 'QTerm'
    T_ORD = 'Traditional'
    ORDERINGS = [Q_ORD, T_ORD]

    NEW_LOSS = 'Improved'
    ORIGINAL_LOSS = 'Original'
    NORMED_LOSS = 'Normed'
    LOSSES = [NEW_LOSS, ORIGINAL_LOSS, NORMED_LOSS]

    @staticmethod
    def valid_options():
        for loss in WaveletDensityEstimator.LOSSES:
            for ordering in WaveletDensityEstimator.ORDERINGS:
                yield loss, ordering


    def calc_pdf_cv(self, xs, loss, ordering, single_threshold=True):
        coeffs = {}
        contributions = []
        alpha_contribution = 0.0
        alpha_norm = 0.0
        for key, tup in self.params.coeffs.items():
            coeff, coeff_b, num = tup
            if coeff == 0.0:
                continue
            j, qx, zs, jpow2 = key
            is_alpha = j == 0 and all([qi == 0 for qi in qx])
            term1, term2, term3, coeff2 = self.params.calc_terms(key, coeff, coeff_b, xs)
            coeff_contribution = term1 - term2 + term3
            if is_alpha:
                coeffs[key] = tup
                alpha_norm += coeff2
                alpha_contribution += coeff_contribution
                continue

            # threshold is the order-by number; here the options
            if loss == WaveletDensityEstimator.ORIGINAL_LOSS:
                if ordering == WaveletDensityEstimator.Q_ORD:
                    threshold = coeff_contribution
                else:
                    threshold = math.fabs(coeff) / math.sqrt(j + 1)
            elif loss == WaveletDensityEstimator.NORMED_LOSS:
                if ordering == WaveletDensityEstimator.T_ORD:
                    threshold = math.fabs(coeff) / math.sqrt(j + 1)
                else:
                    # Hard to justify
                    threshold = coeff_contribution
            else:  # loss == WaveletDensityEstimator.NEW_LOSS:
                if ordering == WaveletDensityEstimator.T_ORD:
                    threshold = math.fabs(coeff) / math.sqrt(j + 1)
                else:
                    threshold = - 0.5 * coeff2 + coeff_contribution

            contributions.append(((key, tup), threshold, (term1, term2, term3, coeff2)))

        if single_threshold:
            self.threshold = self.single_threshold(coeffs, loss, contributions, alpha_norm, alpha_contribution)
            print('Single threshold:', self.threshold.threshold, self.threshold.pos_k, 'params =', len(coeffs))
            return coeffs

        self.thresholds = self.multi_threshold(coeffs, loss, contributions, alpha_norm, alpha_contribution)
        print('Multi-threshold: params =', len(coeffs))
        for threshold in self.thresholds:
            print(threshold.msg, '=', threshold.threshold, threshold.pos_k)
        return coeffs

    def multi_threshold(self, coeffs, loss, contributions, alpha_norm, alpha_contribution):
        current_norm, current_contribution = alpha_norm, alpha_contribution
        thresholds = []
        for j in range(self.delta_j):
            # TODO : remove interacting w/ removed ones; and/or define this 0.1 thingy
            fx = lambda values: values[0][0][0] == j and math.fabs(values[1]) > 0.1/self.params.n
            contributions_j = list(filter(fx, contributions))
            if not contributions_j:
                print('nothing at j=%d' % j)
                continue
            vals_j = self.calc_c_function(contributions_j, current_norm, current_contribution, loss)
            if len(vals_j) == 0:
                continue
            k = self.pos_k_max(vals_j, self.params.n - len(coeffs))
            # import code
            # code.interact(banner='At j=%d' % j, local=locals())
            threshold = ThresholdResult(contributions_j[k][1], k, vals_j, '%d' % j)
            thresholds.append(threshold)
            for values in contributions_j[:k]:
                key, tup = values[0]
                term1, term2, term3, coeff2 = values[2]
                coeff_contribution = term1 - term2 + term3
                current_norm += coeff2
                current_contribution += coeff_contribution
                coeffs[key] = tup
        return thresholds

    def single_threshold(self, coeffs, loss, contributions, alpha_norm, alpha_contribution):
        # self.calc_c_function(contributions, alpha_norm, alpha_contribution)
        if len(coeffs) >= self.params.n:
            # do not go beyond number of observations
            return ThresholdResult(0, -1, np.array([]), 'Too many params')
        vals = self.calc_c_function(contributions, alpha_norm, alpha_contribution, loss)
        if len(vals) == 0:
            return ThresholdResult(0, -1, np.array([]), 'No betas')
        k = self.pos_k_max(vals, self.params.n - len(coeffs))
        for values in contributions[:k]:
            key, tup = values[0]
            coeffs[key] = tup
        return ThresholdResult(contributions[k][1], k, vals, 'All')

    def calc_c_function(self, contributions, curr_norm, curr_contribution, loss):
        contributions.sort(key=lambda values: -values[1])
        print('curr_norm, curr_contribution =', curr_norm, curr_contribution)

        if loss == WaveletDensityEstimator.ORIGINAL_LOSS:
            target_sum = -curr_contribution
        elif loss == WaveletDensityEstimator.NORMED_LOSS:
            target_sum = -curr_contribution
        else: # loss == WaveletDensityEstimator.NEW_LOSS:
            target_sum = 0.5 + 0.5 * curr_norm - curr_contribution

        total_norm = curr_norm
        vals = []
        for values in contributions:
            threshold = values[1]
            term1, term2, term3, coeff2 = values[2]
            total_norm += coeff2
            coeff_contribution = term1 - term2 + term3
            if loss == WaveletDensityEstimator.ORIGINAL_LOSS:
                target_sum += coeff_contribution
                target = 1 - target_sum
            elif loss == WaveletDensityEstimator.NORMED_LOSS:
                target_sum += coeff_contribution
                target = 1 - target_sum / total_norm
            elif loss == WaveletDensityEstimator.NEW_LOSS:
                target_sum += 0.5 * coeff2 - coeff_contribution
                target = target_sum
            else:
                raise ValueError('Unknown loss=%s' % loss)
            vals.append((threshold, 1 - target))
        return np.array(vals)

    def pos_k_max(self, vals, max_params):
        vals_smooth = smooth(vals[:, 1], 5)
        return min(np.argmax(vals_smooth), max_params)

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
