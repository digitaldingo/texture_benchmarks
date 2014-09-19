import os

import joblib
from joblib import Parallel, delayed
from joblib import Memory
memory = Memory(cachedir=os.environ['CACHE_ROOT'],
                verbose=int(os.environ['VERBOSITY']),
                mmap_mode='r',
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

from IPython import embed
from profilehooks import profile


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


#@memory.cache
def compute_dists(model, ref_models):
    if type(model) == joblib.memory.MemorizedResult:
        model = model.get()
    return st.chisquare(model + 1, ref_models + 1, axis=1)[0]



class VarmaZissermanClassifier(BaseEstimator):
    def __init__(self, build_classes, train_classes, n_build=13, n_clusters=10,
                 use_all_classes=False):

        self.build_classes = build_classes
        self.train_classes = train_classes
        self.n_build = n_build
        self.n_clusters = n_clusters
        self.use_all_classes = use_all_classes


    def fit(self, X, y):

        for el in X:
            if type(el) not in (np.ndarray, joblib.memory.MemorizedResult):
                raise TypeError("Input must be a list of ndarrays or "
                                "MemorizedResult instances containing image "
                                "responses.")

        # TODO: Check that build_classes + train_classes (if ints) sum to the
        # total number of classes (or is less).

        # Find build images:
        build_images = {}
        # TODO: Reduce the if statements.
        if type(self.build_classes) == int:
            for i,tclass in enumerate(y):
                if len(build_images.keys()) < self.build_classes:

                    # Ensure the key exists:
                    if len(build_images.setdefault(tclass, [])) < self.n_build:
                        build_images[tclass].append(i)

                elif tclass in build_images.keys() and \
                    len(build_images[tclass]) < self.n_build:
                    build_images[tclass].append(i)

        elif type(self.build_classes) in (list, np.ndarray):
            for i,tclass in enumerate(y):
                if tclass in self.build_classes:

                    # Ensure the key exists:
                    if len(build_images.setdefault(tclass, [])) < self.n_build:
                        build_images[tclass].append(i)

        else:
            raise TypeError("build_classes must be either an integer or a list")



        n_threads = int(os.environ['N_THREADS'])
        verbosity = int(os.environ['VERBOSITY'])
        # FIXME: need to wrap the wrapper like when calculating responses.
        #clustering = memory.cache(self._cluster_responses)
        #tmp_clustering = memory.cache(wrapper).call_and_shelve
        #clustering = tmp_clustering
        textons = Parallel(n_jobs=n_threads, verbose=verbosity)(
            delayed(clustering_wrapper)(self, X[images]) for
            tclass,images in build_images.items())

        ## Set up the multiprocessing pool:
        #pool = mp.Pool()

        #textons = []
        #for tclass,images in build_images.items():
        #    # Do asynchronous parallel clustering.
        #    textons.append(pool.apply_async(self._cluster_responses,
        #                                    [X[images]]))

        #pool.close()
        #pool.join()

        #for t in textons:
        #    self.add_textons(t.get())
        #for t,texton in enumerate(textons):
        #    textons[t] = texton.get()
        #self.textons = textons
        self.textons = np.array([t for texton in textons for t in texton.get()])

        #embed()

        # == Training phase:
        # TODO: switch to joblib
        #pool = mp.Pool()

        model_idx = []
        labels = []
        tm = TextureModel(self.textons)
        # TODO: Reduce the if statements.
        if type(self.train_classes) == int and not self.use_all_classes:
            for i,tclass in enumerate(y):
                if tclass in labels or (len(np.unique(labels)) <
                                        self.train_classes):
                    if tclass not in build_images.keys():
                        #models.append(pool.apply_async(tm.fit, [X[i]]))
                        model_idx.append(i)
                        labels.append(tclass)

        if type(self.train_classes) == int and self.use_all_classes:
            for i,tclass in enumerate(y):
                if tclass in labels or (len(np.unique(labels)) <
                                        self.train_classes):
                    if i not in build_images[tclass]:
                        # Don't include images that were used in the building
                        # phase.
                        model_idx.append(i)
                        labels.append(tclass)

        elif type(self.train_classes) in (list, np.ndarray):
            for i,tclass in enumerate(y):
                if tclass in self.build_classes:
                    if i not in build_images[tclass]:
                        # Don't include images that were used in the building
                        # phase.
                        #models.append(pool.apply_async(tm.fit, [X[i]]))
                        model_idx.append(i)
                        labels.append(tclass)

                elif tclass in self.train_classes:
                    #models.append(pool.apply_async(tm.fit, [X[i]]))
                    model_idx.append(i)
                    labels.append(tclass)

        else:
            raise TypeError("train_classes must be either an integer or a list")

        #embed()
        #stop

        #texture_fit = memory.cache(texture_fit_wrapper)
        models = Parallel(n_jobs=n_threads, verbose=verbosity)(
            delayed(texture_fit_wrapper)(tm, X[idx]) for idx in model_idx)
        #models = np.array([texture_fit(tm, X[idx]) for idx in model_idx])


        #pool.close()
        #pool.join()

        self.models = np.empty((len(models), len(self.textons)))
        for m,model in enumerate(models):
            self.models[m] = model.get()
        #self.models = models
        #embed()

        self.y = np.asarray(labels)


    #@profile
    def predict(self, X):
        """
        Predict the class of an image by finding the nearest model in the
        texton distribution space. Input: Image instance.
        """

        # Instantiate the texture model from the textons:
        tm = TextureModel(self.textons)

        # FIXME: handle the test for type better.
        if len(np.shape(X)) == 2 and type(X[0]) != joblib.memory.MemorizedResult:
            # Generate the model of the new image:
            model = tm.fit(X)

            # Find the nearest learned model:
            dists = st.chisquare(model + 1, self.models + 1)
            nearest = np.argmin(dists)

            # Return the label of the nearest model:
            label = self.y[nearest]

            return label

        elif len(np.shape(X)) == 3 or type(X[0]) == joblib.memory.MemorizedResult:
            # Generate models of the new images:
            n_threads = int(os.environ['N_THREADS'])
            verbosity = int(os.environ['VERBOSITY'])
            #if type(X[0]) == joblib.memory.MemorizedResult:
            #    images = []
            #    for image in X:
            #        images.append(image.get())
            #        del image

            #    models = Parallel(n_jobs=n_threads, verbose=verbosity)(
            #        delayed(cache_texture_fit)(tm, image) for image in images)
            #else:
            #    models = Parallel(n_jobs=n_threads, verbose=verbosity)(
            #        delayed(cache_texture_fit)(tm, image) for image in X)

            #print "Fitting to textures...",
            models = Parallel(n_jobs=n_threads, verbose=verbosity)(
                delayed(texture_fit_wrapper)(tm, image) for image in X)
                #delayed(cache_texture_fit)(tm, image) for image in X)
            #print(" done!")

            # Find the nearest learned model:
            # TODO: joblib
            #dists = []
            #for model in models:
            #    model = model.get()
            #    dists.append([st.chisquare(model + 1, m + 1) for m in self.models])

            #embed()
            #stop

            #print "Computing distances to textures...",
            #dists = Parallel(n_jobs=n_threads, verbose=verbosity)(
            #    delayed(compute_dists)(model, self.models) for model in models)

            dists = np.array([compute_dists(model, self.models) for model in models])
            #embed()
            #print(" done!")
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

        #for i, image in enumerate(images):
        #    if type(image) is  joblib.memory.MemorizedResult:
        #        images[i] = image.get()

        #print(np.shape(images[0]))
        #print(type(images[0]))
        #embed()

        # TODO: reshape to get only 8D responses
        #responses = np.reshape(images,
        #                       (np.shape(images)[0] * np.shape(images)[1],
        #                        np.shape(images)[2]))
        #responses = np.reshape(responses,
        #                       (np.shape(responses)[0] * np.shape(responses)[2]
        #                        * np.shape(responses)[3],
        #                        np.shape(responses)[1]))
        d = responses.shape
        responses = np.rollaxis(responses, 1, 4).reshape((d[0] * d[2] * d[3],
                                                          d[1]))

        # Note: responses needs to be float64 due to a bug in scikit-learn's
        # K Means clustering.
        #responses = responses.astype(np.float64)
        #embed()

        clusters = k_means(n_clusters=self.n_clusters)
        clusters.fit(responses)
        #embed()

        return clusters.cluster_centers_

        #rs = ResponseSpace()
        #rs.add_responses(images)
        #rs.cluster(nclusters=self.nclusters)
        #return rs.clusters

