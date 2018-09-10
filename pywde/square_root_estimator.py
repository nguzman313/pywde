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

    def calc_coeffs(self, xs):
        self.n = xs.shape[0]
        self.ball_tree = BallTree(xs)
        xs_balls = self.calculate_nearest_balls(xs)
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

    # factor for num samples n, dimension dim and nearest index k
    def _calc_factor(self):
        v_unit = (np.pi ** (self.wave.dim / 2.0)) / gamma(self.wave.dim / 2.0 + 1)
        return math.sqrt(v_unit) * (gamma(self.k) / gamma(self.k + 0.5)) / math.sqrt(self.n)

    def calculate_nearest_balls(self, xs):
        k_near_radious = self.ball_tree.query(xs, self.k + 1)[0][:, [-1]]
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
            self.params = []
            k_len = int(xs.shape[0] / cv)
            print('CV N=', xs.shape[0], 'cv=', cv, 'k-fold=', k_len, 'samples')
            for k in range(cv):
                wparams = WParams(self)
                # note: wparams.coeffs.keys [indexes] will be the same because
                # they are calculated based on minx,maxx not the training data
                wparams.calc_coeffs(xs[:-k_len])
                wparams.test = xs[-k_len:]
                self.params.append(wparams)
                xs = np.roll(xs, k_len, axis=0)
                print('Fold', k, 'ready')

    def fit(self, xs):
        "Fit estimator to data. xs is a numpy array of dimension n x d, n = samples, d = dimensions"
        print('Regular estimator')
        t0 = datetime.now()
        self._fitinit(xs)
        self.pdf = self.params.calc_pdf(self.params.coeffs) ##self.calc_pdf()
        self.name = '%s, n=%d, j0=%s, Dj=%d FIT' % (self.wave.name, self.params.n, str(self.jj0), self.delta_j)
        print('secs=', (datetime.now() - t0).total_seconds())

    def mdlfit(self, xs, sorting_fun):
        print('MDL estimator')
        t0 = datetime.now()
        self._fitinit(xs)
        self.calc_pdf_mdl(xs, sorting_fun)
        self.name = '%s, n=%d, j0=%s, Dj=%d MDL' % (self.wave.name, self.params.n, str(self.jj0), self.delta_j)
        print('secs=', (datetime.now() - t0).total_seconds())

    def calc_pdf_mdl(self, xs, sorting_fun):
        all_coeffs = list(self.params.coeffs.items())
        all_coeffs.sort(key=sorting_fun)
        xs_sum = self.params.xs_sum_zeros(xs)
        k = 0
        keys = []
        ranking = []
        best_mdl = best_k = None
        norm_const = 0.0
        for key, tup in all_coeffs[:self.params.n]:
            vals, norm_const = self.params.calc1(key, tup, xs, norm_const)
            xs_sum += vals
            pdf_for_xs = (xs_sum * xs_sum) / norm_const
            j, qx, zs, jpow2 = key
            is_alpha = j == 0 and all([qi == 0 for qi in qx])
            k += 1
            keys.append(key)
            if is_alpha:
                continue
            logLL = - np.log(pdf_for_xs).sum()
            penalty = log_riemann_volume_class(k) - log_riemann_volume_param(k, self.params.n)
            # !!! penalty * 4.0 <-- needs justification; space of params is "doubled" as here is a hemisphere (?)
            # why 4? why not 2^(4/3) or something? related to "number of levels"?
            penalty = penalty * 4.0
            if best_mdl is None or logLL + penalty < best_mdl:
                best_mdl = logLL + penalty
                best_k = len(keys)
            rank_tuple = (k, logLL, penalty, logLL + penalty)
            ranking.append(rank_tuple)
        print('MDL all_params=',len(all_coeffs), ' best k=', best_k, 'best MDL=', best_mdl)
        coeffs = {key:self.params.coeffs[key] for key in keys[:best_k]}
        self.pdf = self.params.calc_pdf(coeffs)
        self.ranking = ranking

    def cv2fit(self, xs, cv):
        print('CV_2 estimator')
        t0 = datetime.now()
        self._fitinit(xs, cv=cv)
        final_keys, ranking = self.calc_pdf_cv2()
        self._fitinit(xs)
        coeffs = {key:self.params.coeffs[key] for key in final_keys}
        self.pdf = self.params.calc_pdf(coeffs)
        self.ranking = ranking
        self.name = '%s, n=%d, j0=%s, Dj=%d CV_2' % (self.wave.name, self.params.n, str(self.jj0), self.delta_j)
        print('secs=', (datetime.now() - t0).total_seconds())

    def calc_pdf_cv2(self):
        cv = len(self.params)
        alphas = []
        ## In CV, all params have the same coeffs structure
        coeff_ixs = list(self.params[0].coeffs.keys())
        coeff_ixs.sort(key=_cv2_key_sort)
        xs_sums = [wparam.xs_sum_zeros(wparam.test) for wparam in self.params]
        norm_const = [0.0] * cv
        logLL = [0.0] * cv
        ## calc baseline for alphas
        final_keys = []
        for key in coeff_ixs:
            j, qx, zs, jpow2 = key
            is_alpha = j == 0 and all([qi == 0 for qi in qx])
            if not is_alpha:
                continue
            final_keys.append(key)
            for k, wparam in enumerate(self.params):
                tup = wparam.coeffs[key]
                vals, new_norm = wparam.calc1(key, tup, wparam.test, norm_const[k])
                norm_const[k] = new_norm
                xs_sums[k] += vals
        for k in range(cv):
            pdf_for_xs = (xs_sums[k] * xs_sums[k]) / norm_const[k]
            logLL[k] = np.log(pdf_for_xs).sum()
        print('logLL for alphas',logLL)
        beta_keys = []
        for key in coeff_ixs:
            j, qx, zs, jpow2 = key
            is_alpha = j == 0 and all([qi == 0 for qi in qx])
            if is_alpha:
                continue
            beta_keys.append(key)
        # simple ordering by size (I could have also beta / \sqrt{j+1} as in other places)
        beta_keys.sort(key=lambda key:sum([math.fabs(wparam.coeffs[key][0]) for wparam in self.params]), reverse=True)
        tries, used = 3, self.params[0].n
        history = []
        while tries > 0 and used > 0:
            curLogLL = sum(logLL)/cv
            history.append((len(history), curLogLL))
            logLLs = {}
            for key in beta_keys:
                j, qx, zs, jpow2 = key
                # compute average LL for key, on temporary accumulators
                betaLogLL = []
                for k, wparam in enumerate(self.params):
                    sum_copy = np.copy(xs_sums[k])
                    tup = wparam.coeffs[key]
                    vals, new_norm = wparam.calc1(key, tup, wparam.test, norm_const[k])
                    sum_copy += vals
                    # !!! change here, use HD - CV
                    pdf_for_xs = (sum_copy * sum_copy) / new_norm
                    betaLogLL.append(np.log(pdf_for_xs).sum())
                logLLs[key] = sum(betaLogLL)/cv
            # find best key
            key = max(logLLs.items(), key=lambda tt:tt[1])[0]
            if logLLs[key] > curLogLL + 0.001: # why 0.001
                print(key, 'better', logLLs[key], curLogLL, 'among', len(beta_keys), '(used', used, ')')
                beta_keys.remove(key)
                final_keys.append(key)
                used -= 1
                # update all CV-current sums for selected beta_{key}
                for k, wparam in enumerate(self.params):
                    tup = wparam.coeffs[key]
                    vals, new_norm = wparam.calc1(key, tup, wparam.test, norm_const[k])
                    xs_sums[k] += vals
                    norm_const[k] = new_norm
                    pdf_for_xs = (xs_sums[k] * xs_sums[k]) / norm_const[k]
                    logLL[k] = np.log(pdf_for_xs).sum()
            else:
                beta_keys.remove(key)
                tries -= 1
                if tries > 0:
                    print('Not better, trying again', tries, 'time(s)')
        return final_keys, history

    def cv_hd1_fit(self, xs, cv, sorting_fun):
        print('CV_HD1 estimator')
        t0 = datetime.now()
        self._fitinit(xs, cv=cv)
        final_keys, ranking = self.calc_pdf_cv_hd1(sorting_fun)
        self._fitinit(xs)
        coeffs = {key:self.params.coeffs[key] for key in final_keys}
        self.pdf = self.params.calc_pdf(coeffs)
        self.ranking = ranking
        self.name = '%s, n=%d, j0=%s, Dj=%d CV(%d).HD1' % (self.wave.name, self.params.n, str(self.jj0), self.delta_j, cv)
        print('secs=', (datetime.now() - t0).total_seconds())

    def calc_pdf_cv_hd1(self, sorting_fun):
        n = self.params[0].n
        cv = len(self.params)
        alphas = []
        ## In CV, all params have the same coeffs structure; we sort based on first CV (for now)
        coeff_ixs = list(self.params[0].coeffs.items())
        coeff_ixs.sort(key=sorting_fun)
        xs_sums = [wparam.xs_sum_zeros(wparam.test) for wparam in self.params]
        norm_const = [0.0] * cv
        # HD^2 = 1 - \int \hat{g} \sqrt{f} ~= 1 - \sum_{x_t \in T} \hat{g}_F \sqrt{f(x_t)}
        hd_term2 = [0.0] * cv
        ## calc baseline for alphas
        final_keys = []
        for key_tup in coeff_ixs:
            key, tup = key_tup
            j, qx, zs, jpow2 = key
            is_alpha = j == 0 and all([qi == 0 for qi in qx])
            if not is_alpha:
                continue
            final_keys.append(key)
            for k, wparam in enumerate(self.params):
                tup = wparam.coeffs[key]
                vals, new_norm = wparam.calc1(key, tup, wparam.test, norm_const[k])
                norm_const[k] = new_norm
                xs_sums[k] += vals
        vol_balls = {}
        for k, wparam in enumerate(self.params):
            vol_balls[k] = wparam.calculate_nearest_balls(wparam.test)
        for k, wparam in enumerate(self.params):
            g_for_xs = np.abs(xs_sums[k]) / math.sqrt(norm_const[k])
            hd_term2[k] = np.asscalar(np.dot(g_for_xs, vol_balls[k]))
        print('hd_term2 for alphas', hd_term2)
        beta_keys = []
        for key_tup in coeff_ixs:
            key, tup = key_tup
            j, qx, zs, jpow2 = key
            is_alpha = j == 0 and all([qi == 0 for qi in qx])
            if is_alpha:
                continue
            beta_keys.append(key)
        # simple ordering by size (I could have also beta / \sqrt{j+1} as in other places)
        # beta_keys.sort(key=lambda key:sum([math.fabs(wparam.coeffs[key][0]) for wparam in self.params]), reverse=True)
        tries, used = 3, self.params[0].n
        history = []
        while tries > 0 and used > 0:
            curHDterm2 = sum(hd_term2)/cv
            history.append((len(history), curHDterm2))
            hdTerm2s = {}
            for key in beta_keys:
                j, qx, zs, jpow2 = key
                # compute average HD term2 for key, on temporary accumulators
                betaHDterm2 = []
                for k, wparam in enumerate(self.params):
                    sum_copy = np.copy(xs_sums[k])
                    tup = wparam.coeffs[key]
                    vals, new_norm = wparam.calc1(key, tup, wparam.test, norm_const[k])
                    sum_copy += vals
                    # !!! change here, use HD - CV
                    g_for_xs = np.abs(sum_copy) / math.sqrt(new_norm)
                    betaHDterm2.append(np.asscalar(np.dot(g_for_xs, vol_balls[k])))
                hdTerm2s[key] = sum(betaHDterm2)/cv
            # find best key; max here is min HD as term2 in HD is negative
            key = max(hdTerm2s.items(), key=lambda tt:tt[1])[0]
            if hdTerm2s[key] > curHDterm2 * (1.0 + 0.1/n): # why 0.001
                print(key, 'better', hdTerm2s[key], curHDterm2, 'among', len(beta_keys), '(used', used, ')')
                beta_keys.remove(key)
                final_keys.append(key)
                used -= 1
                # update all CV-current sums for selected beta_{key}
                for k, wparam in enumerate(self.params):
                    tup = wparam.coeffs[key]
                    vals, new_norm = wparam.calc1(key, tup, wparam.test, norm_const[k])
                    xs_sums[k] += vals
                    norm_const[k] = new_norm
                    g_for_xs = np.abs(xs_sums[k]) / math.sqrt(new_norm)
                    hd_term2[k] = np.asscalar(np.dot(g_for_xs, vol_balls[k]))
            else:
                beta_keys.remove(key)
                tries -= 1
                if tries > 0:
                    print('Not better, trying again', tries, 'time(s)')
        return final_keys, history


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
