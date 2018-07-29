import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as stats
from functools import reduce


from pywde.square_root_estimator import WaveletDensityEstimator

class Beta2D(object):
    def __init__(self, a, b, code='beta'):
        self.dist = stats.beta(a, b)
        self.code = code
        self.dim = 2

    def rvs(self, num):
        return self.dist.rvs((num, 2))

    def pdf(self, grid):
        return self.dist.pdf(grid[0]) * self.dist.pdf(grid[1])


class TruncatedMultiNormalD(object):
    """Truncated mixture or multivariate normal distributions. Dimension is inferred from first $\mu$"""

    def __init__(self, probs, mus, covs, code='mult'):
        self.code = code
        self.probs = probs
        self.dists = [stats.multivariate_normal(mean=mu, cov=cov) for mu, cov in zip(mus, covs)]
        self.dim = len(mus[0])
        z = self._pdf(mise_mesh(self.dim))
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
                i = np.random.choice(np.arange(0,len(self.probs)), p=self.probs)
                yield xvs[i]

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
    else:
        raise NotImplemented('Unknown distribution code [%s]' % code)

# a 2d grid of [0,1], n x n, rendered as array of (n^2,2)
def grid_as_vector(n):
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    return np.meshgrid(x, y)


def plot_wde(wde, fname, title):
    print('Plotting %s' % fname)
    grid_n = 200
    xx, yy = grid_as_vector(grid_n)
    zz = wde.pdf((xx, yy))

    zz_sum = zz.sum() / grid_n / grid_n  # not always near 1

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(xx, yy, zz / zz_sum, edgecolors='k', linewidth=0.5, cmap=cm.get_cmap('BuGn'))
    ax.set_title(wde.name)
    plt.savefig(fname)
    plt.close()
    print('%s saved' % fname)


wde = WaveletDensityEstimator((('db4', 0),('db4', 0)) , k=1, delta_j=3) # bior3.7
dist = dist_from_code('mult')
data = dist.rvs(2000)
print('Estimating')
wde.fit(data)
plot_wde(wde, 'mult-db4-orig.png', wde.name)
print('Estimating with MDL')
wde.mdlfit(data)
plot_wde(wde, 'mult-db4-mdl.png', wde.name)
ranking = np.array(wde.ranking)
print('>> shape: ', ranking.shape)
pos_min = np.argmin(ranking[:,3])
plt.figure(figsize=(10,4))
plt.plot(ranking[:,0], ranking[:,3])
plt.xlabel('# coefficients')
plt.ylabel('MDL')
plt.savefig('mult-mdl-curve.png')
plt.close()
print('Done')