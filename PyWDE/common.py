from __future__ import division
import pywt
import math
import numpy as np
import itertools as itt
from sklearn.neighbors import BallTree
from scipy.interpolate import interp1d
from scipy.special import gamma
from functools import partial

def np_mult(cols):
    if len(cols) == 1:
        return cols[0]
    if len(cols) == 2:
        return np.multiply(*cols)
    else:
        return np.multiply(cols[0], np_mult(cols[1:]))

# all possible combinations of {0,1}, i.e. {0,1}^dim
def all_qx(dim):
    for wave_x, qx in enumerate(itt.product(xrange(2), repeat=dim)):
        yield wave_x, tuple(qx)

def support_tensor(qx, phi_supp, psi_supp):
    return np.array([
        [(phi_supp if q == 0 else psi_supp)[0] for d,q in enumerate(qx)],
        [(phi_supp if q == 0 else psi_supp)[1] for d,q in enumerate(qx)]
    ])
# all z-min, z-max ranges as tensor of ranges for phi and psi support
def all_zs_tensor(zs_min, zs_max):
    it = zip(zs_min, zs_max)
    return [xrange(int(a), int(b)+1) for a, b in it]

# tensor product of z-min per dimension
def z0_tensor(qx, zs_phi, zs_psi):
    return [(zs_phi if q2 == 0 else zs_psi)[0][d] for d, q2 in enumerate(qx)]

# tensor product of max k per dimension
def z1_tensor(qx, zs_phi, zs_psi):
    return [(zs_phi if q2 == 0 else zs_psi)[1][d] for d, q2 in enumerate(qx)]
    
# tensor product waves
# xs = rows x dim
# zs = dim
def wave_tensor(qx, phi, psi, jpow, zs, xs):
    cols = []
    if type(xs) == tuple:
        proj = lambda xs,i: xs[i]
    else:
        proj = lambda xs,i: xs[:,i]
    for i,q2 in enumerate(qx):
        wavef = phi if q2 == 0 else psi
        xs_proj = proj(xs,i)
        cols.append(wavef(jpow * xs_proj - zs[i]))
    return np_mult(tuple(cols)) * (jpow ** (len(qx)/2.0))

def suppf_tensor(qx, phi_sup, psi_sup, jpow, zs, xs):
    cols = []
    if type(xs) == tuple:
        proj = lambda xs,i: xs[i]
    else:
        proj = lambda xs,i: xs[:,i]
    for i,q2 in enumerate(qx):
        supp = phi_sup if q2 == 0 else psi_sup
        xs_proj = proj(xs,i)
        xjz = jpow * xs_proj - zs[i]
        cols.append(np.greater_equal(xjz, supp[0]))
        cols.append(np.less_equal(xjz, supp[1]))
    return np_mult(tuple(cols))


# factor for num samples l, dimension dim and nearest index k
def calc_factor(l, dim, k):
    v_unit = (np.pi ** (dim/2.0)) / gamma(dim/2.0 + 1)
    return math.sqrt(v_unit) * (gamma(k) / gamma(k + 0.5)) / math.sqrt(l)

# calculate V(k);i for each row xs[i] and return dataset with that attached
def calculate_nearest_balls(k, xs):
    dim = xs.shape[1]
    ball_tree = BallTree(xs)
    k_near_radious = ball_tree.query(xs, k + 1)[0][:,[-1]]
    factor = calc_factor(xs.shape[0], dim, k)
    return np.power(k_near_radious, dim/2.0) * factor

def wave_support_info(wave):
    resp = {}
    if wave.family_name in ['Daubechies', 'Symlets']:
        phi_support = (0, wave.dec_len - 1)
        psi_support = (1 - wave.dec_len // 2, wave.dec_len // 2)
        resp['base'] = (phi_support, psi_support)
        resp['dual'] = (phi_support, psi_support)
    elif wave.family_name in ['Coiflets']:
        phi_support = (1 - wave.dec_len // 2, wave.dec_len // 2)
        psi_support = (1 - wave.dec_len // 2, wave.dec_len // 2)
        resp['base'] = (phi_support, psi_support)
        resp['dual'] = (phi_support, psi_support)
    elif wave.family_name == 'Biorthogonal':
        phi_support = (1 - wave.dec_len // 2, wave.dec_len // 2)
        psi_support = (1 - wave.dec_len // 2, wave.dec_len // 2)
        resp['base'] = (phi_support, psi_support)
        raise ValueError('wave family %s not known support' % wave.family_name)
    else:
        raise ValueError('wave family %s not known support' % wave.family_name)
    return resp

def gridify_xs(j0, j1, xs, minx, maxx):
    grid_xs = {}
    dim = xs.shape[1]
    for j in range(j0, j1+1):
        jpow = 2 ** j
        grid_xs[j] = {}
        if j == j0:
            iters = [xrange(int(jpow * minx[d]), int(jpow * maxx[d]) + 1) for d in range(dim)]
            for zs in itt.product(*iters):
                cond = (np.floor(jpow * xs) == zs).all(axis=1)
                grid_xs[j][zs] = np.where(cond)
        else:
            for zs_up, where_xs in grid_xs[j-1].iteritems():
                # TODO theory - one could stop splitting for len <= N0, what does this mean?
                if len(where_xs[0]) == 0:
                    continue
                sub_xs = xs[where_xs]
                zs_up_arr = np.array(zs_up)
                for _, ix2s in all_qx(dim):
                    zs = 2 * zs_up_arr + np.array(ix2s)
                    cond = (np.floor(sub_xs * jpow) == zs).all(axis=1)
                    grid_xs[j][tuple(zs.tolist())] = (where_xs[0][cond],)
    return grid_xs

def zs_range(wavef, minx, maxx, j):
    zs_min = np.floor((2 ** j) * minx - wavef.support[1]) - 1
    zs_max = np.ceil((2 ** j) * maxx - wavef.support[0]) + 1
    return zs_min, zs_max

def calc_coeff(wave_tensor_qx, jpow, zs, xs, xs_balls):
    all_prods = wave_tensor_qx(jpow, zs, xs) * xs_balls[:,0]
    return all_prods.sum()

def calc_num(suppf_tensor_qx, jpow, zs, xs):
    vals = suppf_tensor_qx(jpow, zs, xs)
    return vals.sum()

def calc_coeff_simple(wave_tensor_qx, jpow, zs, xs):
    all_prods = wave_tensor_qx(jpow, zs, xs)
    return all_prods.mean()

