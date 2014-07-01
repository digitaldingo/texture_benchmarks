import os
import numpy as np
import scipy as sp

from skimage.feature import greycomatrix, greycoprops
from skimage.feature import local_binary_pattern
from sklearn.base import BaseEstimator
from joblib import Parallel, delayed

import ipcv
from ipcv import si_hist, josi_hist, bif_hist

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
        self.name = 'bif'

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
    name = 'sih'
    def __init__(self, opts):
        self.opts = opts
        self.name = 'sih'

    def transform(self, X):
        return np.ravel(si_hist(X, **self.opts))


class OrientedShapeIndexHistograms(BaseEstimator):
    name = 'osih'

    def __init__(self, scales, n_bins, tonal_scale, ori_n_bins,
                 ori_tonal_scale, ori_detect_scale, norm, joint_hist,
                 flat=True):
        args = locals().copy()
        del args['self']
        self.__dict__.update(args)

    def transform(self, img):
        hists_shape = (self.n_bins, self.ori_n_bins, len(self.scales))
        hists = np.empty(hists_shape)

        si, si_c, si_o, si_om = ipcv.shape_index(img, self.ori_detect_scale, orientations=True)
        # Detect dominant orientation
        ori_detect_n_bins = self.ori_n_bins*2
        iso_si_o = ipcv.misc.isophotes(si_o, ori_detect_n_bins, (-np.pi/2, np.pi/2),
                             self.ori_tonal_scale, 'von_mises')
        ori_hist = np.sum(iso_si_o, axis=(1,2))
        ori_hist /= np.sum(ori_hist)
#        print(ori_hist)
        max_idx = np.argmax(ori_hist)
        ori_offset = max_idx/float(ori_detect_n_bins)*np.pi-np.pi/2
#        print(max_idx, max_idx/float(ori_detect_n_bins), ori_offset)

        for s_idx, s in enumerate(self.scales):
            si, si_c, si_o, si_om = ipcv.shape_index(img, s, orientations=True)
            # Shift according to dominant orientation
            si_o = np.mod(si_o-ori_offset+np.pi/2, np.pi)-np.pi/2
            # Smooth bin contributions (= soft isophote images)
            iso_si = ipcv.misc.isophotes(si, self.n_bins, (-np.pi/2, np.pi/2), 
                               self.tonal_scale)
            iso_si_o = ipcv.misc.isophotes(si_o, self.ori_n_bins, (-np.pi/2, np.pi/2),
                                 self.ori_tonal_scale, 'von_mises')
            # Bin contributions for the joint histogram
            iso_j = (iso_si[:, np.newaxis, ...] * si_c
                     * iso_si_o[np.newaxis, ...] * si_om)
            # Summarize bin contributions in the joint histograms
            hists[:, :, s_idx] = np.sum(iso_j, axis=(2, 3))
        if not self.joint_hist:
            hists_si = np.sum(hists, axis=1)
            hists_ori = np.sum(hists, axis=0)
            hists_si = ipcv.misc.normalize(hists_si, self.norm)
            hists_ori = ipcv.misc.normalize(hists_ori, self.norm)
            hists = np.hstack([np.ravel(hists_si), np.ravel(hists_ori)])
        else:
            hists = ipcv.misc.normalize(hists, self.norm)
        if self.flat:
            hists = np.ravel(hists)
        return hists


def _transform_one(estimator, X):
    return estimator.transform(X)


class ParallelEstimator(BaseEstimator):
    def __init__(self, estimator):
        self.estimator = estimator
        self.name = estimator.name

    def transform(self, X):
        n_threads = int(os.environ['N_THREADS'])
        verbosity = int(os.environ['VERBOSITY'])
        features = Parallel(n_jobs=n_threads, verbose=verbosity)(
            delayed(_transform_one)(self.estimator, X[i])
            for i in range(X.shape[0]))
        features = np.array(features)
        return features



def intensity_hist(img, scale, n_bins, tonal_scale):
    if np.min(img) < 0 or np.max(img) > 1:
        raise ValueError('Invalid image intensities; should be in [0,1].')
    img = ipcv.scalespace(img, scale)
    limits = (0, 1)
    img_iso = ipcv.misc.isophotes(img, n_bins, limits, tonal_scale, 'gaussian')
    hist = np.sum(img_iso, axis=(1,2))    
    hist /= np.sum(hist)
    return hist


def shape_index_hist(img, scale, n_bins, tonal_scale):
    si, si_c = ipcv.shape_index(img, scale, orientations=False)
#    sp.misc.imsave('woop%.2f.png' % scale, si)
    limits = (-np.pi/2, np.pi/2)
    si_iso = ipcv.misc.isophotes(si, n_bins, limits, tonal_scale)
    hist = np.sum(si_iso * si_c, axis=(1,2))
    hist /= np.sum(hist)
    return hist


def gradient_hist(img, scale, n_bins, tonal_scale):
    go, go_m = ipcv.gradient_orientation(img, scale)
#    sp.misc.imsave('woop%.2f.png' % scale, si)
    limits = (-np.pi/2, np.pi/2)
    go_iso = ipcv.misc.isophotes(go, n_bins, limits, tonal_scale)
    hist = np.sum(go_iso * go_m, axis=(1,2))
    hist /= np.sum(hist)
    return hist

def entropy(hist, alpha):
    if alpha == 1.0:
        return -np.sum(hist * np.log(hist))
    else:
        return 1/(1.0-alpha)*np.log(np.sum(hist**alpha))
