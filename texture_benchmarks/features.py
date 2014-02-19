import os
import numpy as np

from skimage.feature import greycomatrix, greycoprops
from skimage.feature import local_binary_pattern
from sklearn.base import BaseEstimator
from joblib import Parallel, delayed

from ipcv import si_hist, bif_hist


class Haralick(BaseEstimator):
    def __init__(self, opts):
        self.opts = opts

    def transform(self, X):
        return self.haralick(X, **self.opts)

    def haralick(self, img, ndistances, min_distance, dist_diff, nangles,
                 levels, symmetric, normed, features):
        img -= np.min(img)
        img /= np.max(img)
        img_ = np.round(img*(levels-1)).astype(int)
        distances = np.arange(ndistances)*dist_diff + min_distance
        angles = np.linspace(0, np.pi, num=nangles, endpoint=False)
        M = greycomatrix(img_, distances=distances, angles=angles,
                         levels=levels, symmetric=symmetric, normed=normed)
        f = [greycoprops(M, prop=prop) for prop in features]
        return np.ravel(f)


class BasicImageFeatures(BaseEstimator):
    def __init__(self, opts):
        self.opts = opts

    def transform(self, X):
        return bif_hist(X, **self.opts)


class LBP(BaseEstimator):
    def __init__(self, opts):
        self.opts = opts

    def transform(self, X):
        return self.lbp(X, **self.opts)

    def lbp(self, img, nscales, min_scale, nangles, method='uniform'):
        hists = []
        radii = [min_scale+i*2 for i in range(nscales)]
        for r in radii:
            lbp_img = local_binary_pattern(img, nangles, r, method)
            n_bins = nangles+1
            h, _ = np.histogram(lbp_img, normed=True, bins=n_bins,
                                range=(0, n_bins))
            hists.append(h)
        return np.hstack(hists).astype(float)


class ShapeIndexHistograms(BaseEstimator):
    def __init__(self, opts):
        self.opts = opts

    def transform(self, X):
        return np.ravel(si_hist(X, **self.opts))


def _transform_one(estimator, X):
    return estimator.transform(X)


class ParallelEstimator(BaseEstimator):
    def __init__(self, estimator):
        self.estimator = estimator

    def transform(self, X):
        n_threads = int(os.environ['N_THREADS'])
        verbosity = int(os.environ['VERBOSITY'])
        features = Parallel(n_jobs=n_threads, verbose=verbosity)(
            delayed(_transform_one)(self.estimator, X[i])
            for i in range(X.shape[0]))
        features = np.array(features)
        return features
