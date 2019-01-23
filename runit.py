import numpy as np
import math
import random
#import matplotlib.pyplot as plt
import itertools as itt
import csv
#from matplotlib import cm
#from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as stats
import scipy.integrate as integrate
from functools import reduce
#from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
from statsmodels.nonparametric.kernel_density import KDEMultivariate


from pywde.square_root_estimator import WaveletDensityEstimator, coeff_sort, coeff_sort_no_j

import click


def sign(p1, p2, p3):
    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])


def in_triangle(pt, points):
    v1, v2, v3 = points
    b1 = sign(pt, v1, v2) < 0.0
    b2 = sign(pt, v2, v3) < 0.0
    b3 = sign(pt, v3, v1) < 0.0
    return (b1 == b2) and (b2 == b3)

def triangle_area(points):
    a, b, c = points
    return math.fabs(a[0]*b[1] + b[0]*c[1] + c[0]*a[1] - a[0]*c[1] - b[0]*a[1] - c[0]*b[1])/2.0


def _pdf(probs, dists, grid):
    assert len(probs) == len(dists)
    if type(grid) == tuple or type(grid) == list:
        pos = np.stack(grid, axis=0)
        pos = np.moveaxis(pos, 0, -1)
    else:
        pos = grid
    pdf_vals = None
    for prob, dist in zip(probs, dists):
        vals = dist.pdf(pos)
        if pdf_vals is None:
            pdf_vals = vals * prob
        else:
            pdf_vals = np.add(pdf_vals, vals * prob)
    return pdf_vals


class PyramidDist(object):
    def __init__(self, points, centre, code='pyr1'):
        if not in_triangle(centre, points):
            raise ValueError("centre must be inside")
        self.code = code
        self.dim = 2
        vol = triangle_area(points) / 3
        self.height = 1.0 / vol
        pp = [(x, y, 0) for x,y in points]
        #pp += [(0,0,0),(0,1,0),(1,0,0),(1,1,0)]
        pp += [(centre[0], centre[1], self.height)]
        pp = np.array(pp)
        self.fun = LinearNDInterpolator(pp[:,0:2], pp[:,2], fill_value=0.0)

    def rvs(self, num):
        data = []
        while num > 0:
            pp = (random.random(), random.random())
            u = random.random()
            if u <= max(0,self.fun(*pp)) / self.height:
                data.append(pp)
                num -= 1
                if num == 0:
                    break
        return np.array(data)

    def pdf(self, grid):
        if type(grid) == tuple or type(grid) == list:
            X, Y = grid
            grid = np.array((X.flatten(), Y.flatten())).T
            vals = np.clip(self.fun(grid),0,None)
            vals = vals.reshape(X.shape[0], Y.shape[0])
            return vals
        else:
            return np.clip(self.fun(grid),0,None)

class TriangleDist(object):
    def __init__(self, points, code='tri1'):
        self.points = points
        self.code = code
        self.dim = 2
        self._h = 1/triangle_area(points)

    def rvs(self, num):
        data = []
        while num > 0:
            pp = (random.random(), random.random())
            if in_triangle(pp, self.points):
                data.append(pp)
                num -= 1
                if num == 0:
                    break
        return np.array(data)

    def pdf(self, grid):
        @np.vectorize
        def inside(x, y):
            return in_triangle((x,y), self.points)
        return np.where(inside(*grid), self._h, 0.0)

class Beta2D(object):
    def __init__(self, a, b, code='beta'):
        self.dist = stats.beta(a, b)
        self.code = code
        self.dim = 2

    def rvs(self, num):
        data = []
        while num > 0:
            for d in self._rvs():
                if 0 <= d[0] and d[0] <= 1 and 0 <= d[1] and d[1] <= 1:
                    data.append(d)
                    num -= 1
                    if num == 0:
                        break
        return np.array(data)

    def pdf(self, grid):
        return self.dist.pdf(grid[0]) * self.dist.pdf(grid[1])


class UniformDistribution(object):
    def __init__(self):
        self.code = 'unif'
        self.dim = 2

    def rvs(self, num):
        return 0.25 + np.random.uniform(size=(num, self.dim))/2

    def pdf(self, grid):
        if type(grid) == tuple or type(grid) == list:
            x, y = grid
        else:
            x = grid[:,0]
            y = grid[:, 1]
        vals = np.less(0.25, x) & np.less(x, 0.75) & np.less(0.25, y) & np.less(y, 0.75)
        return 4 * vals


class TruncatedMultiNormalD(object):
    """Truncated mixture or multivariate normal distributions. Dimension is inferred from first $\mu$"""

    def __init__(self, probs, mus, covs, code='mult'):
        self.code = code
        self.probs = probs
        self.dists = [stats.multivariate_normal(mean=mu, cov=cov) for mu, cov in zip(mus, covs)]
        self.dim = len(mus[0])
        z = _pdf(self.probs, self.dists, mise_mesh(self.dim))
        nns = reduce(lambda x, y: (x-1) * (y-1), z.shape)
        self.sum = z.sum()/nns

    def mathematica(self):
        # render Mathematica code to plot
        def fn(norm_dist):
            mu = np.array2string(norm_dist.mean, separator=',')
            mu = mu.replace('[','{').replace(']','}').replace('e','*^')
            cov = np.array2string(norm_dist.cov, separator=',')
            cov = cov.replace('[','{').replace(']','}').replace('e','*^')
            return 'MultinormalDistribution[%s,%s]' % (mu, cov)
        probs = '{%s}' % ','.join([str(f / min(self.probs)) for f in self.probs])
        dists = '{%s}' % ','.join([fn(d) for d in self.dists])
        resp = 'MixtureDistribution[%s,%s]' % (probs, dists)
        return resp

    def _rvs(self):
        while True:
            for xvs in zip(*[dist.rvs(100) for dist in self.dists]):
                yield xvs

    def rvs(self, num):
        data = []
        while num > 0:
            for dd in self._rvs():
                i = np.random.choice(np.arange(0,len(self.probs)), p=self.probs)
                d = dd[i]
                if 0 <= d[0] and d[0] <= 1 and 0 <= d[1] and d[1] <= 1:
                    data.append(d)
                    num -= 1
                    if num == 0:
                        break
        return np.array(data)

    def _pdf(self, grid):
        if type(grid) == tuple or type(grid) == list:
            pos = np.stack(grid, axis=0)
            pos = np.moveaxis(pos, 0, -1)
        else:
            pos = grid
        vals = [dist.pdf(pos) for dist in self.dists]
        pdf_vals = vals[0] * self.probs[0]
        for i in range(len(self.probs) - 1):
            pdf_vals = np.add(pdf_vals, vals[i+1] * self.probs[i+1])
        #pdf_vals = pdf_vals / total
        return pdf_vals

    def pdf(self, grid):
        return self._pdf(grid)/self.sum


class TruncatedLaplace2D(object):
    def __init__(self, mu, scale, code='lap1', angle=30.):
        self.mu = mu
        self.scale = scale
        self.dim = 2
        self.code = code
        theta = (angle / 180.) * np.pi
        self.rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
        self.sum = self._pdfsum()

    def rvs(self, num):
        resp = None
        while num > 0:
            vs = np.random.laplace(0.0, scale=self.scale, size=(num + 10, 2))
            vs = np.matmul(vs, self.rot) + self.mu
            cond = (0 <= vs[:, 0]) & (vs[:, 0] <= 1.0) & (0 <= vs[:, 1]) & (vs[:, 1] <= 1)
            vs = vs[cond][:num]
            if len(vs) == 0:
                continue
            num -= vs.shape[0]
            if resp is not None:
                resp = np.concatenate((resp, vs))
            else:
                resp = vs
        return resp

    def pdf(self, grid):
        if type(grid) == tuple or type(grid) == list:
            pos = np.stack(grid, axis=0)
            vals = np.moveaxis(pos, 0, -1)
            print('new shape', vals.shape)
        else:
            vals = grid
        # print('vals', vals)
        # print('vals - mu', vals - self.mu)
        # print('(vals - mu) rot^T', np.matmul(vals - self.mu, self.rot.T))
        vals = np.matmul(vals - self.mu, self.rot.T)
        vals = np.exp(-abs(vals)/self.scale) / (2 * self.scale)
        #print('pdf 2', vals)
        if len(vals.shape) == 2:
            vals = vals[:,0] * vals[:,1]
        else:
            vals = vals[:,:,0] * vals[:,:,1]
        return vals / self.sum

    def _pdfsum(self):
        grid = mise_mesh(self.dim)
        pos = np.stack(grid, axis=0)
        vals = np.moveaxis(pos, 0, -1)
        vals = np.matmul(vals - self.mu, self.rot.T)
        resp = np.exp(-abs(vals)/self.scale) / (2 * self.scale)
        xx, yy = resp[:,:,0], resp[:,:,1]
        nns = reduce(lambda x, y: x * y, resp.shape)/resp.shape[-1]
        resp = (xx * yy).sum()/nns
        return resp


class MixtureDistribution(object):
    def __init__(self, probs, dists, code=None):
        self.probs = probs
        self.dists = dists
        self.dim = dists[0].dim
        assert all([self.dim == dist.dim for dist in dists])
        if code:
            self.code = code
        else:
            self.code = 'x'.join([dist.code for dist in dists])
        z = _pdf(self.probs, self.dists, mise_mesh(self.dim))
        nns = reduce(lambda x, y: (x-1) * (y-1), z.shape)
        self.sum = z.sum()/nns

    def pdf(self, grid):
        return _pdf(self.probs, self.dists, grid)/self.sum

    def _rvs(self):
        while True:
            for xvs in zip(*[dist.rvs(100) for dist in self.dists]):
                yield xvs

    def rvs(self, num):
        data = []
        while num > 0:
            for dd in self._rvs():
                i = np.random.choice(np.arange(0,len(self.probs)), p=self.probs)
                d = dd[i]
                if 0 <= d[0] and d[0] <= 1 and 0 <= d[1] and d[1] <= 1:
                    data.append(d)
                    num -= 1
                    if num == 0:
                        break
        return np.array(data)


def mise_mesh(d=2):
    grid_n = 256 if d == 2 else 40
    VVs = [np.linspace(0.0,1.0, num=grid_n) for i in range(d)]
    return np.meshgrid(*VVs)


def dist_from_code(code):
    if code == 'beta':
        return Beta2D(2, 4, code=code)
    elif code == 'mult' or code == 'mul2':
        sigma = 0.01
        return TruncatedMultiNormalD(
            [1.5/9, 7.5/9],
            [np.array([0.2, 0.3]), np.array([0.7, 0.7])],
            [np.array([[sigma/6, 0], [0, sigma/6]]), np.array([[0.015, sigma/64], [sigma/64, 0.015]])],
            code=code
        )
    elif code == 'mul3':
        sigma = 0.01
        return TruncatedMultiNormalD(
            [0.4, 0.3, 0.3],
            [np.array([0.3, 0.4, 0.35]),
             np.array([0.7, 0.7, 0.6]),
             np.array([0.7, 0.6, 0.35])],
            [np.array([[0.02, 0.01, 0.], [0.01, 0.02, 0.], [0., 0., 0.02]]),
             np.array([[0.0133333, 0., 0.], [0., 0.0133333, 0.], [0., 0., 0.0133333]]),
             np.array([[0.025, 0., 0.], [0., 0.025, 0.01], [0., 0.01, 0.025]])
             ],
            code=code
            )
    elif code == 'mix1':
        sigma = 0.05
        m1 = np.array([[sigma/6, 0], [0, sigma/6.5]])
        return TruncatedMultiNormalD(
            [1/2, 1/2],
            [np.array([0.2, 0.3]), np.array([0.7, 0.7])],
            [m1, m1],
            code=code
            )
    elif code == 'mix2':
        sigma = 0.05
        angle = 10.
        theta = (angle/180.) * np.pi
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta),  np.cos(theta)]])
        m1 = np.array([[sigma/6, 0], [0, sigma/8]])
        m2 = np.dot(rot, np.dot(m1, rot.T))
        return TruncatedMultiNormalD(
            [1/2, 1/2],
            [np.array([0.4, 0.3]), np.array([0.7, 0.7])],
            [m1, m2],
            code=code
            )
    elif code == 'mix3':
        sigma = 0.03
        angle = 10.
        theta = (angle/180.) * np.pi
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta),  np.cos(theta)]])
        m1 = np.array([[sigma/6, 0], [0, sigma/7]])
        m2 = np.dot(rot, np.dot(m1, rot.T))
        prop = np.array([8,4,2,1])
        prop = prop/prop.sum()
        return TruncatedMultiNormalD(
            prop.tolist(),
            [np.array([0.2, 0.3]), np.array([0.5, 0.5]), np.array([0.65, 0.7]), np.array([0.82, 0.85])],
            [m1, m2/2, m1/4, m2/8],
            code=code
            )
    elif code == 'mix4':
        sigma = 0.03
        angle = 10.
        theta = (angle / 180.) * np.pi
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
        m1 = np.array([[sigma / 6, 0], [0, sigma / 7]])
        m2 = np.dot(rot, np.dot(m1, rot.T))
        prop = np.array([8, 4, 2, 1, 384])
        prop = prop / prop.sum()
        return TruncatedMultiNormalD(
            prop.tolist(),
            [np.array([0.2, 0.3]), np.array([0.5, 0.5]), np.array([0.65, 0.7]), np.array([0.82, 0.85]), np.array([0.5, 0.5])],
            [m1, m2 / 2, m1 / 4, m2 / 8, 0.18 * np.eye(2, 2)],
            code=code
        )
    elif code == 'mix5':
        sigma = 0.03
        angle = 10.
        theta = (angle / 180.) * np.pi
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
        m1 = np.array([[sigma / 6, 0], [0, sigma / 7]])
        m2 = np.dot(rot, np.dot(m1, rot.T))
        prop = np.array([8, 4, 2, 1])
        prop = prop / prop.sum()
        return TruncatedMultiNormalD(
            prop.tolist(),
            [np.array([0.2, 0.3]), np.array([0.5, 0.5]), np.array([0.65, 0.7]), np.array([0.82, 0.85]),
             np.array([0.5, 0.5])],
            [m1, m2 / 2, m1 / 6, m2 / 8],
            code=code
        )
    elif code == 'mix6':
        theta = np.pi / 4
        rot = lambda angle : np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        m0 = np.array([[0.1, 0], [0, 0.0025]])
        m1 = np.dot(rot(theta), np.dot(m0, rot(theta).T)) / 2
        m2 = np.dot(rot(-theta), np.dot(m0, rot(-theta).T)) / 2
        prop = np.array([1, 1])
        prop = prop / prop.sum()
        return TruncatedMultiNormalD(
            prop.tolist(),
            [np.array([0.3, 0.3]), np.array([0.7, 0.3])], [m1, m2],
            code=code
        )
    elif code == 'mix7': ## not good
        m0 = np.array([[0.1, 0], [0, 0.005]])
        m1 = np.array([[0.005, 0], [0, 0.1]])
        prop = np.array([1, 1, 1, 1])
        prop = prop / prop.sum()
        return TruncatedMultiNormalD(
            prop.tolist(),
            [np.array([0.5, 0.3]),
             np.array([0.5, 0.7]),
             np.array([0.3, 0.5]),
             np.array([0.7, 0.5])],
            [m0, m0, m1, m1],
            code=code
        )
    elif code == 'mix8':
        sigma = 0.03
        angle = 10.
        theta = (angle / 180.) * np.pi
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
        m1 = np.array([[sigma / 6, 0], [0, sigma / 8]])
        m2 = np.dot(rot, np.dot(m1, rot.T))
        prop = np.array([8, 6, 5, 3])
        prop = prop / prop.sum()
        return TruncatedMultiNormalD(
            prop.tolist(),
            [np.array([0.2, 0.3]), np.array([0.5, 0.5]), np.array([0.65, 0.7]), np.array([0.82, 0.85]),
             np.array([0.5, 0.5])],
            [m1, m2 / 1.5, m1 / 2, m2 / 3],
            code=code
        )
    elif code == 'mix9':
        sigma = 0.03
        angle = 10.
        theta = (angle / 180.) * np.pi
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
        m1 = np.array([[sigma / 6, 0], [0, sigma / 8]])
        m2 = np.dot(rot, np.dot(m1, rot.T))
        prop = np.array([5, 50, 1, 1])
        prop = prop / prop.sum()
        return TruncatedMultiNormalD(
            prop.tolist(),
            [np.array([0.2, 0.3]), np.array([0.5, 0.5]), np.array([0.65, 0.7]), np.array([0.82, 0.85])],
            [m1, np.array([[0.05,0],[0,0.05]]), m2 / 4.5, m1 / 5],
            code=code
        )
    elif code == 'lap1':
        return TruncatedLaplace2D(np.array([0.5, 0.5]), 0.1, code)
    elif code == 'lap2':
        return TruncatedLaplace2D(np.array([0.5, 0.5]), 0.1, code, angle=45.)
    elif code == 'lap3':
        return TruncatedLaplace2D(np.array([0.4, 0.4]), 0.1, code, angle=45.)
    elif code == 'tri1':
        return TriangleDist(((0.1,0.2),(0.3,0.7),(0.8,0.2)))
    elif code == 'pyr1':
        return PyramidDist(((0.1,0.2),(0.4,0.9),(0.8,0.2)), (0.4,0.3))
    elif code == 'pyr2':
        return PyramidDist(((0.1, 0.2), (0.4, 0.9), (0.8, 0.2)), (0.4, 0.3))
    elif code == 'unif':
        return UniformDistribution()
    else:
        raise NotImplemented('Unknown distribution code [%s]' % code)

# a 2d grid of [0,1], n x n, rendered as array of (n^2,2)
def grid_as_vector(n):
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    return np.meshgrid(x, y)


def calc_maxv(dist):
    grid_n = 70
    xx, yy = grid_as_vector(grid_n)
    zz = dist.pdf((xx, yy))
    print('sum=', zz.mean())
    zz_sum = zz.sum() / grid_n / grid_n  # not always near 1
    print('int =', zz_sum)
    return (zz / zz_sum).max()

def plot_dist(fname, dist):
    grid_n = 70
    xx, yy = grid_as_vector(grid_n)
    zz = dist.pdf((xx, yy))
    print('sum=', zz.mean())
    zz_sum = zz.sum() / grid_n / grid_n  # not always near 1
    print('int =', zz_sum)
    max_v = (zz / zz_sum).max()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(xx, yy, zz / zz_sum, edgecolors='k', linewidth=0.5, cmap=cm.get_cmap('BuGn'))
    ax.set_title(dist.code)
    ax.set_zlim(0, 1.1 * max_v)
    #plt.show()
    plt.savefig(fname)
    plt.close()
    print('%s saved' % fname)

def plot_wde(wde, fname, dist, zlim):
    print('Plotting %s' % fname)
    hd, corr_factor = hellinger_distance(dist, wde)
    print(wde.name, 'HD=', hd)
    ##return
    grid_n = 40 ## 70
    xx, yy = grid_as_vector(grid_n)
    zz = wde.pdf((xx, yy)) / corr_factor

    ##zz_sum = zz.sum() / grid_n / grid_n  # not always near 1

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(xx, yy, zz, edgecolors='k', linewidth=0.5, cmap=cm.get_cmap('BuGn'))
    ax.set_title(wde.name + ('\nHD = %g' % hd))
    ax.set_zlim(0, zlim)
    plt.savefig(fname)
    plt.close()
    print('%s saved' % fname)


def plot_energy(wde, fname):
    fig = plt.figure()
    xx = wde.vals[:,0]
    yy = wde.vals[:,1]
    plt.plot(xx, yy)
    plt.axvline(x=wde.threshold, c='r')
    plt.ylim(min(yy)*0.95, max(yy)*1.05)
    plt.xlabel('C')
    plt.ylabel('$B_C$')
    plt.savefig(fname)
    plt.close()
    print('%s saved' % fname)
    fname = fname.replace('energy', 'energy2')
    fig = plt.figure()
    xx = range(wde.vals.shape[0])
    yy = wde.vals[:,1]
    plt.plot(xx, yy)
    plt.axvline(x=wde.pos_k, c='r')
    plt.ylim(min(yy)*0.95, max(yy)*1.05)
    plt.xlabel('$i$')
    plt.ylabel('$B_i$')
    plt.savefig(fname)
    plt.close()
    print('%s saved' % fname)


def plot_kde(kde, fname, dist, zlim):
    print('Plotting %s' % fname)
    hd, corr_factor = hellinger_distance(dist, kde)
    print('kde HD=', hd)
    grid_n = 40 ## 70
    xx, yy = grid_as_vector(grid_n)
    grid2 = np.array((xx.flatten(), yy.flatten())).T
    vals = kde.pdf(grid2)
    zz = vals.reshape(xx.shape[0], yy.shape[0])

    ##zz_sum = zz.sum() / grid_n / grid_n  # not always near 1

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(xx, yy, zz, edgecolors='k', linewidth=0.5, cmap=cm.get_cmap('BuGn'))
    ax.set_title(('KDE bw=%s' % str(kde.bw)) + ('\nHD = %g' % hd))
    ax.set_zlim(0, zlim)
    plt.savefig(fname)
    plt.close()
    print('%s saved' % fname)


def hellinger_distance_wip(dist, dist_est):
    def ferr(x, y):
        args = np.array([(x, y)])
        pdf_vals = np.sqrt(dist.pdf(args))
        est_vals = np.sqrt(dist_est.pdf(args))/corr_factor
        return ((pdf_vals - est_vals) ** 2)[0]
    def pdf(x, y):
        est_vals = dist_est.pdf(np.array([(x, y)]))
        return est_vals[0]
    corr_factor = integrate.dblquad(pdf, 0.0, 1.0, lambda x:0.0, lambda x:1.0)
    err = integrate.dblquad(ferr, 0.0, 1.0, lambda x: 0.0, lambda x: 1.0)
    return err, corr_factor


def hellinger_distance(dist, dist_est):
    grid = grid_as_vector(256)
    pdf_vals = dist.pdf(grid)
    #print('DIST:', pdf_vals.mean())
    pdf_vals = pdf_vals / pdf_vals.mean()
    pdf_vals = np.sqrt(pdf_vals)
    if isinstance(dist_est, KDEMultivariate):
        X, Y = grid
        grid2 = np.array((X.flatten(), Y.flatten())).T
        vals = dist_est.pdf(grid2)
        pred_vals = vals.reshape(X.shape[0], Y.shape[0])
    else:
        pred_vals = dist_est.pdf(grid)
    #print('WDE:', pred_vals.mean())
    corr_factor = pred_vals.mean()
    print('corr factor = %g' % corr_factor)
    pred_vals = pred_vals / corr_factor
    pred_vals = np.sqrt(pred_vals)
    diff = pdf_vals - pred_vals
    err = (diff * diff).mean()  ## !!! /2
    return err, corr_factor


def fname(what, dist_name, num=None, wave_name=None, delta_j=None, ext='.png'):
    strn = '%04d' % num if num is not None else None
    strd = '%d' % delta_j if delta_j is not None else None
    strs = [dist_name, what, strn, wave_name, strd]
    strs = [v for v in strs if v]
    return 'pngs/%s%s' % ('-'.join(strs), ext)


def save_data(data, fname):
    with open(fname, 'wt') as fh:
        writer = csv.writer(fh, delimiter='\t')
        for row in data:
            writer.writerow(row)

@click.group()
def main():
    pass

@main.command()
@click.argument('dist_name', metavar="DIST_CODE")
def plot_true(dist_name):
    dist = dist_from_code(dist_name)
    plot_dist(fname('true', dist_name), dist)


@main.command()
@click.argument('dist_name', metavar="DIST_CODE")
@click.argument('wave_name', metavar="WAVE_CODE")
@click.argument('num', type=int)
@click.argument('ix', type=int, nargs=2)
@click.argument('delta_js', nargs=-1, type=int)
# @click.option('--loss', help='Loss function', default=WaveletDensityEstimator.NEW_LOSS)
# @click.option('--ordering', help='Ordering method', default=WaveletDensityEstimator.T_ORD)
# @click.option('--k', type=int, default=1)
def run_with(dist_name, wave_name, num, ix, delta_js):
    dest = fname('results', dist_name, ext='-%02d.tab' % ix[0])
    with open(dest, 'wt') as fh:
        writer = csv.writer(fh, delimiter='\t')
        i0, numi = ix
        for row in calc_with(dist_name, wave_name, num, i0, numi, delta_js):
            writer.writerow(row)
            fh.flush()

def calc_with(dist_name, wave_name, num, i0, numi, delta_js):
    dist = dist_from_code(dist_name)
    ## max_v = calc_maxv(dist)
    yield ['dist', 'wave', 'num', 'sample_num', 'method', 'k', 'delta_j', 'loss', 'ordering', 'HD']
    for ix in range(numi):
        data = dist.rvs(num)
        i = i0 + ix
        save_data(data, fname('data', dist_name, num=num, wave_name=wave_name, ext='(%02d).csv' % i))
        print('===== %02d =====' % i)
        for k, delta_j in itt.product([1,2], delta_js):
            wde = WaveletDensityEstimator(((wave_name, 0),(wave_name, 0)) , k=k, delta_j=delta_j)
            print('WDE', 'k=%d' % k, 'delta_j=%d' % delta_j)
            wde.fit(data)
            hd, corr_factor = hellinger_distance(dist, wde)
            yield [dist_name, wave_name, num, i, 'WDE', k, delta_j, '', '', hd]
            ## plot_wde(wde, fname('orig', dist_name, num, wave_name, delta_j), dist, 1.1 * max_v)
            for loss, ordering in WaveletDensityEstimator.valid_options():
                print('WDE', 'k=%d' % k, 'delta_j=%d' % delta_j, 'Loss', loss, 'Ord', ordering)
                wde.cvfit(data, loss, ordering)
                hd, corr_factor = hellinger_distance(dist, wde)
                yield [dist_name, wave_name, num, i, 'WDE_CV', k, delta_j, loss, ordering, hd]
                # what = 'new_%s.%s' % (loss, ordering)
                # plot_wde(wde, fname(what, dist_name, num, wave_name, delta_j), dist, 1.1 * max_v)
                # what = 'energy_%s.%s' % (loss, ordering)
                # plot_energy(wde, fname(what, dist_name, num, wave_name, delta_j))
        #print('Estimating KDE all data')
        kde = KDEMultivariate(data, 'c' * data.shape[1], bw='cv_ml') ## cv_ml
        hd, corr_factor = hellinger_distance(dist, kde)
        yield [dist_name, wave_name, num, i, 'KDE', '', '', '', '', hd]
        # plot_kde(kde, fname('kde_cv', dist_name, num), dist, 1.1 * max_v)


#
# TODO - check consitency bby increasing sample size !!!
#
# - chicken, cai - look
# - why 3 - double check formula and algorithm
# - then send again to Spiro & Gery
# - find better distributions to showcase
#

#dist = dist_from_code('tri1')
#print(dist.rvs(10))
#plot_dist('tri1.png', dist)
# dist = dist_from_code('pir1')
# data = dist.rvs(1024)
# plt.figure()
# plt.scatter(data[:,0], data[:,1])
# plt.show()


if __name__ == "__main__":
    main()
