#
# helpers
#
import numpy as np
import math

def assert_integral(dim, f1, f2, exp_value, prec):
    a1, b1 = map(np.array, f1.support)
    a2, b2 = map(np.array, f2.support)
    a = np.amin(np.stack((a1,a2)), axis=0)
    b = np.amax(np.stack((b1,b2)), axis=0)
    if dim == 1:
        val1 = f1(np.linspace(a, b, num=int(10000*(b-a))))
        val2 = f2(np.linspace(a, b, num=int(10000*(b-a))))
        assert_almost_equal(exp_value, (val1 * val2).sum()/10000.0, prec)
    else:
        xx = np.stack((a, b)).T
        x0 = np.linspace(*xx[0], num=int(300*(xx[0,1] - xx[0,0]) + 0.5))
        x1 = np.linspace(*xx[1], num=int(300*(xx[1,1] - xx[1,0]) + 0.5))
        x0, x1 = np.meshgrid(x0, x1)
        val1 = f1((x0, x1))
        val2 = f2((x0, x1))
        assert_almost_equal(exp_value, (val1 * val2).sum()/90000.0, prec)


def assert_almost_equal(float1, float2, prec):
    msg = 'Expected %f, Actual %f (to %d digits)' % (float1, float2, prec)
    fact10 = 10 ** prec
    assert int(math.fabs(float1 - float2) * fact10) == 0, msg


def intersect_1d(intval1, intval2):
    a, b = intval1
    x, y = intval2
    return x <= b and a <= y


def intersect_2d(region1, region2):
    return (intersect_1d((region1[0][0], region1[1][0]), (region2[0][0], region2[1][0])) and
            intersect_1d((region1[0][1], region1[1][1]), (region2[0][1], region2[1][1])))
