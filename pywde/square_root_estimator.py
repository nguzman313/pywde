import math
import numpy as np
import itertools as itt
from .common import all_zs_tensor
from sklearn.neighbors import BallTree
from scipy.special import gamma
from datetime import datetime

from .pywt_ext import WaveletTensorProduct

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

    def calc_coeffs(self, xs, cv=False):
        self.n = xs.shape[0]
        self.ball_tree = BallTree(xs)
        self.calculate_nearest_balls(xs, cv)
        norm = 0.0
        omega = self.omega(self.n)
        for key in self.coeffs.keys():
            j, qx, zs, jpow2 = key
            jpow2 = np.array(jpow2)
            num = self.wave.supp_ix('dual', (qx, jpow2, zs))(xs).sum()
            terms = self.wave.fun_ix('dual', (qx, jpow2, zs))(xs)
            coeff = (terms * self.xs_balls).sum() * omega
            #print('beta_{%s}' % str(key), '=', coeff)
            self.coeffs[key] = (coeff, num)
            norm += coeff * coeff
        print('calc_coeffs #', len(self.coeffs), norm)

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
        omega_n1 = self.omega(self.n-1)
        omega_n2 = omega_n * omega_n1

        coeff_dual = (fun_i_base * self.xs_balls).sum() * omega_n
        term1 = omega_n1 / omega_n * coeff * coeff_dual

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
        return term1, term2, term3, coeff * coeff_dual

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
            zs_min, zs_max = self.wave.z_range('dual', (qx, jpow2, None), self.minx, self.maxx)
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
        self.name = '%s, n=%d, j0=%s, Dj=%d FIT #params=%d' % (self.wave.name, self.params.n, str(self.jj0),
                                                               self.delta_j, len(self.params.coeffs))
        print('secs=', (datetime.now() - t0).total_seconds())

    def cvfit(self, xs):
        print('CV estimator')
        t0 = datetime.now()
        self._fitinit(xs, cv=True)
        coeffs = self.calc_pdf_cv(xs)
        self.pdf = self.params.calc_pdf(coeffs)
        self.name = '%s, n=%d, j0=%s, Dj=%d CV new #params=%d' % (self.wave.name, self.params.n, str(self.jj0),
                                                              self.delta_j, len(coeffs))
        print('secs=', (datetime.now() - t0).total_seconds())

    def calc_pdf_cv(self, xs):
        coeffs = {}
        contributions = []
        alpha_contribution = 0.0
        alpha_norm = 0.0
        i = 0
        for key_tup in self.params.coeffs.items():
            key, tup = key_tup
            coeff, num = tup
            if coeff == 0.0:
                continue
            j, qx, zs, jpow2 = key
            is_alpha = j == 0 and all([qi == 0 for qi in qx])
            term1, term2, term3, coeff2 = self.params.calc_terms(key, coeff, xs)
            coeff_contribution = term1 - term2 + term3
            if is_alpha:
                i += 1
                coeffs[key] = tup
                alpha_norm += coeff2
                alpha_contribution += coeff_contribution
                continue
            threshold = math.fabs(coeff) / math.sqrt(j + 1)
            #threshold = 2 * coeff_contribution * (1 + coeff_contribution)
            # threshold = math.fabs(coeff2 - coeff_contribution) ## <<- up and downs in run_with('mix9', 'db4', 500, 3)
            #threshold = math.fabs(coeff2 - coeff_contribution)
            #threshold = coeff2 - coeff_contribution
            contributions.append(((key, tup), threshold, (term1, term2, term3, coeff2)))
            #print(contributions[-1])
        contributions.sort(key=lambda values: -values[1])
        print('alpha_norm, alpha_contribution =', alpha_norm, alpha_contribution)
        target = 0.5 + 0.5 * alpha_norm - alpha_contribution
        total_norm = alpha_norm
        total_i = i
        #min_v = 0.01/self.params.n
        #print('min_v',max_v)
        self.vals = []
        for values in contributions:
            threshold = values[1]
            term1, term2, term3, coeff2 = values[2]
            total_norm += coeff2
            coeff_contribution = term1 - term2 + term3
            target += 0.5 * coeff2 - coeff_contribution
            key, tup = values[0]
            ## print(key, coeff2, coeff_contribution, 'tots : ', total_norm, total_contribution) ## << print Q
            ## self.vals.append((threshold, total_contribution))
            ## print(key, target)
            self.vals.append((threshold, 1 - target))
            i += 1
        self.vals = np.array(self.vals)
        approach = 'max'
        if approach == 'max':
            vals = smooth(self.vals[:,1], 5)
            k = np.argmax(vals)
            print('argmax=', k)
            # no more than number of points min(k, self.params.n - total_i)
            pos_k = min(k, self.params.n - total_i)
            self.threshold = contributions[k][1]
            self.pos_k = pos_k
        elif approach == 'min':
            vals = smooth(self.vals, 5)
            k = np.argmin(vals)
            print('argmin=', k)
            # no more than number of points min(k, self.params.n - total_i)
            pos_k = min(k, self.params.n - total_i)
        else: # close to 1
            vals = np.array(self.vals)
            pos_neg = np.argmax(vals > min(max(vals) - 0.001, 1))
            print('vals @ pos_neg', pos_neg, vals[max(0,pos_neg - 3): pos_neg + 3])
            # qs = np.array([tripple[1] for tripple in contributions])
            # pos_neg = np.argmax(qs < 0.0) - 1
            # print('qs[..]=',qs[pos_neg-3:pos_neg+3])
            # sum = 0.0
            # while sum < 0.01 and pos_neg >= 0:
            #     sum += qs[pos_neg]
            #     pos_neg -= 1
            # print('to-1 pos', pos_neg)
            pos_k = min(pos_neg, self.params.n - total_i)
        for values in contributions[:pos_k]:
            key, tup = values[0]
            coeffs[key] = tup
        # remove adjacent blocks >> does not work, too much
        # dz = int(math.log(self.params.n)/2 + 0.5)
        # for tripple in contributions[pos_k:]:
        #     key, tup = tripple[0]
        #     j, qx, zs, jpow2 = key
        #     for z0 in range(-dz,dz+1):
        #         for z1 in range(-dz,dz+1):
        #             zn = (zs[0] + z0, zs[1] + z1)
        #             nkey = (j, qx, zn, jpow2)
        #             if nkey in coeffs:
        #                 del coeffs[nkey]
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
