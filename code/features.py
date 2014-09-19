import os
import numpy as np
import scipy as sp

from skimage.feature import greycomatrix, greycoprops
from skimage.feature import local_binary_pattern
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import preprocessing
import joblib
from joblib import Parallel, delayed

import ipcv
from ipcv import si_hist, josi_hist, bif_hist

from ipcv.filters import StackedFilters, GaussianFilter, LOGFilter,\
                         EdgeFilter, BarFilter

from profilehooks import profile

from IPython import embed


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
        features = Parallel(n_jobs=n_threads, verbose=verbosity,
                            #pre_dispatch='3*n_jobs', temp_folder=None
                            )( delayed(_transform_one)(self.estimator, X[i]) for
                              i in range(X.shape[0]))
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


# -----------------------------------------------------------------------------


#class GaussianFilter(BaseEstimator):
#    """
#    A Gaussian filter.
#    """
#    def __init__(self, sigma, order=0, mode="constant", cval=0):
#        self.sigma = sigma
#        self.order = order
#        self.mode = mode
#        self.cval = cval
#
#    def fit(self, X, y=None):
#        """
#        Do nothing and return the estimator unchanged.
#
#        This method is just there to implement the usual API and hence work in
#        pipelines.
#        """
#        return self
#
#    def transform(self, X, y=None):
#
#        fsize = int(np.max(self.sigma) * 5)
#        if fsize % 2 == 0:
#            fsize += 1
#        impulse = np.zeros([fsize,fsize])
#        impulse[int(fsize / 2),int(fsize / 2)] = 1
#        self.filter_ = nd.gaussian_filter(impulse, sigma=self.sigma,
#                                         order=self.order, mode=self.mode,
#                                         cval=self.cval)
#
#        return nd.convolve(X, self.filter_)
#
#
#class LOGFilter(GaussianFilter):
#    """
#    A Laplacian of Gaussian (LOG) filter.
#    """
#    def __init__(self, sigma, mode="constant", cval=0):
#        super().__init__(sigma=sigma, order=2, mode=mode, cval=cval)
#
#
#class AnisotropicGaussianFilter(BaseEstimator):
#    """
#    An isotropic Gaussian filter.
#    """
#    def __init__(self, sigma, order, angle=0, mode="mirror", cval=0):
#        self.sigma = sigma
#        self.order = order
#        self.angle = angle
#        self.mode = mode
#        self.cval = cval
#
#    def fit(self, X, y=None):
#        """
#        Do nothing and return the estimator unchanged.
#
#        This method is just there to implement the usual API and hence work in
#        pipelines.
#        """
#        return self
#
#    def transform(self, X, y=None, factor=5):
#        fsize = int(np.max(self.sigma) * factor)
#        if fsize % 2 == 0:
#            fsize += 1
#        impulse = np.zeros([fsize,fsize])
#        impulse[int(fsize / 2),int(fsize / 2)] = 1
#        f = nd.gaussian_filter(impulse, sigma=self.sigma, order=self.order,
#                               mode=self.mode, cval=self.cval)
#        self.filter = nd.interpolation.rotate(f, self.angle, reshape=False)
#
#        return nd.convolve(X, self.filter)
#
#
#class EdgeFilter(AnisotropicGaussianFilter):
#    """
#    An edge filter based on an isotropic Gaussian filter.
#    """
#    def __init__(self, sigma, order=(1,0), angle=0, **kwargs):
#        super().__init__(sigma=sigma, order=order, angle=angle, **kwargs)
#
#
#
#class BarFilter(AnisotropicGaussianFilter):
#    """
#    A bar filter based on an isotropic Gaussian filter.
#    """
#    def __init__(self, sigma, order=(2,0), angle=0, **kwargs):
#        super().__init__(sigma=sigma, order=order, angle=angle, **kwargs)



def bla(f, x):
    return f.apply(x)

class MR8FilterBank(BaseEstimator, TransformerMixin):
    """
    Obtain responses from an MR8 filter bank.
    """
    def __init__(self, sigma=10, scales=((1,3), (2,6), (4,12)),
                 angles=(0,30,60,90,120,150)):

        self.sigma = sigma
        self.scales = scales
        self.angles = angles

        self.name = "MR8 filter bank"

    def fit(self, X, y=None):
        pass


    def transform(self, X, y=None):
        # TODO: remove all dependencies on Image container.

        bank = StackedFilters()

        bank.add_filter(GaussianFilter(self.sigma))
        bank.add_filter(LOGFilter(self.sigma))

        for scale in self.scales:
            sf = StackedFilters()
            for angle in self.angles:
                sf.add_filter(EdgeFilter(scale, angle=angle))
            bank.add_filter(sf)

        for scale in self.scales:
            sf = StackedFilters()
            for angle in self.angles:
                sf.add_filter(BarFilter(scale, angle=angle))
            bank.add_filter(sf)

        # Normalise all filters:
        bank.normalise()

        #verbosity = int(os.environ['VERBOSITY'])
        if len(np.shape(X)) == 2:
            # Normalise image:
            # FIXME: should scaling be applied as a separate preprocessing step?
            X = preprocessing.scale(X)

            responses = bank.apply(X).responses

        elif len(np.shape(X)) == 3:
            # Normalise images:
            # FIXME: should scaling be applied as a separate preprocessing step?
            X = [preprocessing.scale(img) for img in X]

            responses = [bank.apply(img).responses for img
                        in X]

        else:
            raise ValueError("Input needs to be either two or three "
                             "dimensional.")

        #responses = Parallel(n_jobs=self.n_jobs, verbose=verbosity,
        #                     max_nbytes="10M")(delayed(bla)(bank, img) for
        #                                       img in X)


        if len(np.shape(X)) == 2:
            for r,res in enumerate(responses):
                if res.__class__.__name__ == "StackedFilters":
                    # Get maximum response:
                    responses[r] = res.maximum_response

        elif len(np.shape(X)) == 3:
            for i,result in enumerate(responses):
                responses[i] = result.responses
                for r,res in enumerate(result.responses):
                    if res.__class__.__name__ == "StackedFilters":
                        # Get maximum response:
                        result.responses[r] = res.maximum_response

        # Convert to numpy array:
        self.responses_ = np.asarray(responses)

        #embed()

        #return self.responses_

        # Magical contrast normalisation:
        L2 = np.sqrt(np.sum(self.responses_**2, axis=0))
        self.responses_ *= np.log(1 + L2/0.03)/L2

        return self.responses_


def distance(textons, responses):
    return np.linalg.norm(textons - responses, axis=1)

def compute_dists(textons, responses):
    dists = np.empty((len(responses), len(textons)))

    for i,texton in enumerate(textons):
        dists[:,i] = np.linalg.norm(responses - texton, axis=1)
    return dists


class TextureModel(BaseEstimator):
    """
    Model a texture as a histogram of pixel distances to textons.
    """

    def __init__(self, textons):
        self.textons = textons


    def fit(self, X, y=None):
        """
        Calculate a histogram for a texture.
        """

        if type(X) == joblib.memory.MemorizedResult:
            responses = X.get()
            del X
        else:
            responses = X

        ## TODO: reshape to get only 8D responses
        #responses = np.reshape(X,
        #                       (np.shape(images)[0] * np.shape(images)[1],
        #                        np.shape(images)[2]))
        d = responses.shape
        responses = np.rollaxis(responses, 0, 3).reshape((d[1] * d[2], d[0]))


        # For each pixel response, find the nearest texton:
        dists = np.empty((len(responses), len(self.textons)))
        ## TODO: joblib
        #for i,response in enumerate(responses):
        #    dists[i] = np.linalg.norm(self.textons - response, axis=1)

        for i,texton in enumerate(self.textons):
            dists[:,i] = np.linalg.norm(responses - texton, axis=1)

        #dists = compute_dists(self.textons, responses)

        #dists = np.linalg.norm(self.textons[:,None,:] - np.tile(responses,
        #                                                        (200,1,1)),
        #                       axis=2)

        #n_threads = int(os.environ['N_THREADS'])
        #verbosity = int(os.environ['VERBOSITY'])
        #dists = Parallel(n_jobs=n_threads, verbose=verbosity)(
        #    delayed(distance)(self.textons, r) for r in responses)


        #embed()
        nearest = np.argmin(dists, axis=1)
        #nearest = np.argmin(dists, axis=0)

        # Create a histogram of the frequencies of the nearest textons. This
        # will be the model.
        hist = np.bincount(nearest, minlength=np.shape(self.textons)[0])

        return hist


