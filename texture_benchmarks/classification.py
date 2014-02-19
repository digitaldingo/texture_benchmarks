import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import chi2_kernel, additive_chi2_kernel

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline


def bhattacharyya_kernel(x, y):
    return (1 - np.dot(np.sqrt(x), np.sqrt(y.T)))**2


class BIF_KNNClassifier(KNeighborsClassifier):
    def __init__(self, k=5):
        super(BIF_KNNClassifier, self).__init__(n_neighbors=k,
                                                metric=chi2_kernel)


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
