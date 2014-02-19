#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np

import data
import features
import classification

import logging
logging.basicConfig(level=logging.INFO)

from joblib import Memory
memory = Memory(cachedir=os.environ['CACHE_ROOT'],
                verbose=int(os.environ['VERBOSITY']))
#memory.clear()


def cross_validate(dataset, feat_ex, classifier):
    print('# Loading dataset')
    imgs = memory.cache(dataset.imgs)()

    print('# Feature description')
    features = memory.cache(feat_ex.transform)(imgs)
    print('features.shape:', features.shape)

    print('# Classification')
    labels = dataset.labels
    splits = dataset.splits()
    split_ids = range(len(splits))
    accuracies = []
    for k, (train_idx, test_idx) in zip(split_ids, splits):
        x_train = features[train_idx, :]
        y_train = labels[train_idx]
        x_test = features[test_idx, :]
        y_test = labels[test_idx]
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        accuracy = np.sum(y_pred == y_test) / float(y_test.size)
        accuracies.append(accuracy)
    mean = np.mean(accuracies)
    std = np.std(accuracies)
    print('%s, avg. accuracy: %.4f +/- %.4f' % (dataset.name, mean, std))
    return mean


if __name__ == '__main__':
    # Select dataset
#    dataset = data.CUReTGray()
    dataset = data.KTH_TIPS()
#    dataset = data.UIUCTex()

    # Select feature extractor
#    opts_bif = {
#        'n_scales': 4,
#        'scale_min': 1.0,
#        'scale_ratio': 2.0,
#        'norm': 'l1',
#    }
#    feat_ex = features.BasicImageFeatures(opts_bif)
    opts_sih = {
        'n_scales': 4,
        'scale_min': 1.0,
        'scale_ratio': 1.8,
        'n_bins': 16,
        'tonal_scale': .2,
        'norm': 'l1',
    }
    feat_ex = features.ShapeIndexHistograms(opts_sih)
    feat_ex = features.ParallelEstimator(feat_ex)

    # Select classifier
    classifier = classification.SVMClassifier(C=50.0)
#    classifier = classification.BIF_KNNClassifier()

    # Select classifier
    cross_validate(dataset, feat_ex, classifier)
