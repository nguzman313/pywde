import pywt

class SimpleSquareRoot(object):
    def __init__(self, wave_name, k=1, j0=1, j1=None, thresholding=None):
        self.wave = pywt.Wavelet(wave_name)
        self.k = k
        self.j0 = j0
        self.j1 = j1 if j1 is not None else (j0 - 1)
        self.multi_supports = wave_support_info(self.wave)
        self.pdf = None
        if thresholding is None:
            self.thresholding = lambda n, j, dn, c: c
        else:
            self.thresholding = thresholding
