import math
import itertools as itt
import numpy as np
from collections import namedtuple
from scipy.special import gamma
from sklearn.neighbors import BallTree

from pywde.pywt_ext import WaveletTensorProduct
from pywde.common import all_zs_tensor


class SPWDE(object):
    def __init__(self, waves, k=1):
        self.wave = WaveletTensorProduct([wave_desc[0] for wave_desc in waves])
        self.j0s = [wave_desc[1] for wave_desc in waves]
        self.k = k
        self.minx = None
        self.maxx = None

    def best_j(self, xs):
        balls_info = calc_sqrt_vs(xs, self.k)
        self.minx = np.amin(xs, axis=0)
        self.maxx = np.amax(xs, axis=0)
        for j in range(9):
            # calc B hat
            tots = []
            self.calc_funs(j, xs)
            for i, x in enumerate(xs):
                alphas = self.calc_alphas_no_i(j, xs, i, balls_info, self.dual_fun)
                norm2 = 0.0
                g_ring_x = 0.0
                for zs in alphas:
                    alpha_zs = alphas[zs]
                    g_ring_x += alpha_zs * self.base_fun[zs][i]
                    # todo: only orthogonal case
                    norm2 += alpha_zs * alpha_zs
                # q_ring_x ^ 2 / norm2 == f_at_x
                if norm2 == 0.0:
                    if g_ring_x == 0.0:
                        tots.append(0.0)
                    else:
                        raise RuntimeError('Got norms but no value')
                else:
                    tots.append(g_ring_x * g_ring_x /  norm2)
            tots = np.array(tots)
            b_hat_j = calc_omega(xs.shape[0], self.k) * (np.sqrt(tots) * balls_info.sqrt_vol_k).sum()
            print(j, b_hat_j)

    def calc_funs(self, j, xs):
        jj = [j + j0 for j0 in self.j0s]
        jpow2 = np.array([2 ** j for j in jj])

        qq = (0, 0)  # alphas
        funs = {}
        for what in ['dual', 'base']:
            zs_min, zs_max = self.wave.z_range(what, (qq, jj, None), self.minx, self.maxx)
            funs[what] = {}
            for zs in itt.product(*all_zs_tensor(zs_min, zs_max)):
                funs[what][zs] = self.wave.fun_ix(what, (qq, jpow2, zs))(xs)
        self.base_fun = funs['base']
        self.dual_fun = funs['dual']


    def calc_alphas_no_i(self, j, xs, i, balls_info, dual_fun):
        qq = (0, 0) # alphas
        jj = [j + j0 for j0 in self.j0s]
        zs_min, zs_max = self.wave.z_range('dual', (qq, jj, None), self.minx, self.maxx)
        omega_no_i = calc_omega(xs.shape[0] - 1, self.k)
        resp = {}
        balls = balls_no_i(balls_info, i)
        for zs in itt.product(*all_zs_tensor(zs_min, zs_max)):
            # below, we remove factor for i from sum << this has the most impact in performance
            alpha_zs = omega_no_i * ((dual_fun[zs] * balls).sum() - dual_fun[zs][i] * balls[i])
            resp[zs] = alpha_zs
        return resp


def balls_no_i(balls_info, i):
    n = balls_info.nn_indexes.shape[0]
    resp = []
    for i_prim in range(n):
        # note index i is removed at callers site
        if i in balls_info.nn_indexes[i_prim, :-1]:
            resp.append(balls_info.sqrt_vol_k_plus_1[i_prim])
        else:
            resp.append(balls_info.sqrt_vol_k[i_prim])
    return np.array(resp)


def calc_omega(n, k):
    "Bias correction for k-th nearest neighbours sum for sample size n"
    return math.sqrt(n - 1) * gamma(k) / gamma(k + 0.5) / n


BallsInfo = namedtuple('BallsInfo', ['sqrt_vol_k', 'sqrt_vol_k_plus_1', 'nn_indexes'])


def calc_sqrt_vs(xs, k):
    "Returns BallsInfo object with sqrt of volumes of k-th balls and (k+1)-th balls"
    dim = xs.shape[1]
    ball_tree = BallTree(xs)
    # as xs is both data and query, xs's nearest neighbour would be xs itself, hence the k+2 below
    dist, inx = ball_tree.query(xs, k + 2)
    k_near_radious = dist[:, -2:]
    xs_balls_both = np.power(k_near_radious, dim / 2)
    xs_balls = xs_balls_both[:, 0] * sqrt_vunit(dim)
    xs_balls2 = xs_balls_both[:, 1] * sqrt_vunit(dim)
    return BallsInfo(xs_balls, xs_balls2, inx)


def sqrt_vunit(dim):
    "Square root of Volume of unit hypersphere in d dimensions"
    return math.sqrt((np.pi ** (dim / 2)) / gamma(dim / 2 + 1))
