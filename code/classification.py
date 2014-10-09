import os

import joblib
from joblib import Parallel, delayed, Memory
memory = Memory(cachedir=os.environ['CACHE_ROOT'],
                verbose=int(os.environ['VERBOSITY']),
                )

import numpy as np
import scipy.stats as st

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics.pairwise import chi2_kernel, additive_chi2_kernel

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.cluster import MiniBatchKMeans as k_means

from features import TextureModel


def bhattacharyya(x, y):
    return (1 - np.dot(np.sqrt(x), np.sqrt(y.T)))**2


class SVMClassifier(BaseEstimator):
    def __init__(self, C=1.0, kernel='rbf', gamma=0.0, class_weight='auto',
                 tol=1e-3):
        pipeline_steps = []

        # Feature normalization
        if kernel == 'rbf':
            with_mean = True
        else:
            with_mean = False
        self.scaler = StandardScaler(with_mean=with_mean)
        pipeline_steps.append(('scaler', self.scaler))

        # Feature classification
        if kernel == 'chi2':
            kernel = chi2_kernel
        elif kernel == 'additive_chi2':
            kernel = additive_chi2_kernel
        svm = SVC(C=C, kernel=kernel, gamma=gamma, class_weight=class_weight,
                  tol=1e-5)

        pipeline_steps.append(('svm', svm))

        self.pipeline = Pipeline(pipeline_steps)

    def fit(self, X, y=None):
        self.pipeline.fit(X, y)
        return self

    def transform(self, X, y=None, copy=None):
        return self.pipeline.transform(X, y, copy)

    def predict(self, X):
        return self.pipeline.predict(X)


@memory.cache
def cache_clustering(method, X):
    return method._cluster_responses(X)


def clustering_wrapper(method, X):
    return cache_clustering.call_and_shelve(method, X)


@memory.cache
def cache_texture_fit(texture_model, X):
    return texture_model.fit(X)


def texture_fit_wrapper(texture_model, X):
    return cache_texture_fit.call_and_shelve(texture_model, X)


def compute_dists(model, ref_models):
    if type(model) == joblib.memory.MemorizedResult:
        model = model.get()
    return st.chisquare(model + 1, ref_models + 1, axis=1)[0]



class VarmaZissermanClassifier(BaseEstimator):
    def __init__(self, build_classes=None, train_classes=None, n_build=13,
                 n_clusters=10, use_all_classes=False):

        self.build_classes = build_classes
        self.train_classes = train_classes
        self.n_build = n_build
        self.n_clusters = n_clusters
        self.use_all_classes = use_all_classes


    def fit(self, X, y):

        # ========================= Building phase =========================

        for el in X:
            if type(el) not in (np.ndarray, joblib.memory.MemorizedResult):
                raise TypeError("Input must be a list of ndarrays or "
                                "MemorizedResult instances containing image "
                                "responses.")

        if self.build_classes is None:
            self.build_classes = np.unique(y)

        if self.train_classes is None:
            self.train_classes = np.unique(y)

        # Find images for dictionary building:
        build_images = {}
        # TODO: Reduce the if statements.
        if type(self.build_classes) == int:
            for i,tclass in enumerate(y):
                if len(build_images.keys()) < self.build_classes:

                    # Ensure the key exists and add the image if we haven't
                    # reached the number of building images for that class:
                    if len(build_images.setdefault(tclass, [])) < self.n_build:
                        build_images[tclass].append(i)

                elif tclass in build_images.keys() and \
                        len(build_images[tclass]) < self.n_build:
                    build_images[tclass].append(i)

        elif type(self.build_classes) in (list, np.ndarray):
            for i,tclass in enumerate(y):
                if tclass in self.build_classes:

                    # Ensure the key exists and add the image if we haven't
                    # reached the number of building images for that class:
                    if len(build_images.setdefault(tclass, [])) < self.n_build:
                        build_images[tclass].append(i)

        else:
            raise TypeError("build_classes must be either an integer or a list")



        n_threads = int(os.environ['N_THREADS'])
        verbosity = int(os.environ['VERBOSITY'])
        textons = Parallel(n_jobs=n_threads, verbose=verbosity)(
            delayed(clustering_wrapper)(self, X[images]) for
            tclass,images in build_images.items())

        self.textons = np.array([t for texton in textons for t in texton.get()])


        # ========================= Training phase =========================

        model_idx = []
        labels = []
        tm = TextureModel(self.textons)

        # TODO: Reduce the if statements.
        if type(self.train_classes) == int and not self.use_all_classes:
            for i,tclass in enumerate(y):
                if tclass in labels or (len(np.unique(labels)) <
                                        self.train_classes):
                    if tclass not in build_images.keys():
                        model_idx.append(i)
                        labels.append(tclass)

        elif type(self.train_classes) == int and self.use_all_classes:
            for i,tclass in enumerate(y):
                if tclass in labels or (len(np.unique(labels)) <
                                        self.train_classes):
                    if i not in build_images[tclass]:
                        # Don't include images that were used in the building
                        # phase.
                        # FIXME: is this the intended behaviour?
                        model_idx.append(i)
                        labels.append(tclass)

        elif type(self.train_classes) in (list, np.ndarray):
            for i,tclass in enumerate(y):
                if tclass in self.build_classes:
                    if i not in build_images[tclass]:
                        # Don't include images that were used in the building
                        # phase.
                        # FIXME: is this the intended behaviour?
                        model_idx.append(i)
                        labels.append(tclass)

                elif tclass in self.train_classes:
                    model_idx.append(i)
                    labels.append(tclass)

        else:
            raise TypeError("train_classes must be either an integer or a list")

        models = Parallel(n_jobs=n_threads, verbose=verbosity)(
            delayed(texture_fit_wrapper)(tm, X[idx]) for idx in model_idx)


        self.models = np.empty((len(models), len(self.textons)))
        for m,model in enumerate(models):
            self.models[m] = model.get()

        self.y = np.asarray(labels)


    def predict(self, X):
        """
        Predict the class of an image by finding the nearest model in the
        texton distribution space. Input: Image instance.
        """

        # Instantiate the texture model from the textons:
        tm = TextureModel(self.textons)

        # FIXME: handle the test for type better.
        #if len(np.shape(X)) == 2:# and type(X[0]) != joblib.memory.MemorizedResult:
        if len(np.shape(X)) == 2 and all([type(x) !=
                                          joblib.memory.MemorizedResult for x in
                                          X]):

            # Generate the model of the new image:
            model = tm.fit(X)

            # Find the nearest learned model:
            dists = st.chisquare(model + 1, self.models + 1)
            nearest = np.argmin(dists)

            # Return the label of the nearest model:
            label = self.y[nearest]

            return label

        #elif len(np.shape(X)) == 3 or type(X[0]) == joblib.memory.MemorizedResult:
        elif len(np.shape(X)) == 3 or any([type(x) ==
                                           joblib.memory.MemorizedResult for x
                                           in X]):

            # Generate models of the new images:
            n_threads = int(os.environ['N_THREADS'])
            verbosity = int(os.environ['VERBOSITY'])

            models = Parallel(n_jobs=n_threads, verbose=verbosity)(
                delayed(texture_fit_wrapper)(tm, image) for image in X)

            # Find the nearest learned model:
            dists = np.array([compute_dists(model, self.models) for model in
                              models])
            nearest = np.argmin(dists, axis=1)

            # Return the labels of the nearest models:
            labels = self.y[nearest]

            return labels



    def _cluster_responses(self, images):
        """
        Do a K Means clustering of image responses.
        """

        responses = np.array([im.get() if type(im) ==
                              joblib.memory.MemorizedResult else im for im in
                              images], dtype=np.float64)
        del images


        d = responses.shape
        responses = np.rollaxis(responses, 1, 4).reshape((d[0] * d[2] * d[3],
                                                          d[1]))

        clusters = k_means(n_clusters=self.n_clusters)
        clusters.fit(responses)

        return clusters.cluster_centers_
