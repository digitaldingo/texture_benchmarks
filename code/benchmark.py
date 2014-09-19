#!/usr/bin/env python
# coding: utf-8

import os
os.environ['CACHE_ROOT'] = '../cache'
os.environ['TEXTURE_DATA_ROOT'] = '../data'
os.environ['RESULTS_ROOT'] = '../results'
os.environ['VERBOSITY'] = '10'
os.environ['N_THREADS'] = '3'

import numpy as np

import data
import features
import classification
import plots

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
    #feats = memory.cache(feat_ex.transform)(imgs)
    feats = memory.cache(feat_ex.fit_transform)(imgs)
    print('Feature dimensions:', feats.shape[1])

    print('# Classification')
    labels = dataset.labels
    splits = dataset.splits()
    split_ids = range(len(splits))
    accuracies = []
    y_preds = []
    y_tests = []
    idxs = []
    for k, (train_idx, test_idx) in zip(split_ids, splits):
        x_train = feats[train_idx, :]
        y_train = labels[train_idx]
        x_test = feats[test_idx, :]
        y_test = labels[test_idx]
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        accuracy = np.sum(y_pred == y_test) / float(y_test.size)
        accuracies.append(accuracy)
        y_preds.append(y_pred)
        y_tests.append(y_test)
        idxs.append(test_idx)
    mean = np.mean(accuracies)
    std = np.std(accuracies)
    plots.difficult_imgs(dataset, feat_ex, feats, idxs, y_tests, y_preds)
    print('%s, avg. accuracy: %.4f +/- %.4f' % (dataset.name, mean, std))
    return mean


def run():
    # Select dataset
    dataset = data.CUReTGray()
#    dataset = data.KTH_TIPS()
#    dataset = data.UIUCTex()

    # Select feature extractor
    opts_bif = {
        'n_scales': 4,
        'scale_min': 1.0,
        'scale_ratio': 2.0,
        'norm': 'l1',
    }
    opts_sih = {
        'scales': 1*2.0**np.arange(5),
        'n_bins': 24,
        'tonal_scale': .2,
        'norm': 'l1',
    }

#    feat_ex = features.BasicImageFeatures(opts_bif)
    feat_ex = features.ShapeIndexHistograms(opts_sih)

    feat_ex = features.OrientedShapeIndexHistograms(**{
        'scales': 1*2.0**np.arange(5),
        'n_bins': 18,
        'tonal_scale': .2,
        'ori_n_bins': 8,
        'ori_tonal_scale': 0.25,
        'ori_detect_scale': 4,
        'norm': 'l1',
        'joint_hist': True,
    })
    feat_ex = features.ParallelEstimator(feat_ex)

    feat_ex = features.MR8FilterBank()

    # Select classifier
    classifier = classification.SVMClassifier(C=200, kernel='rbf')
#    classifier = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1,
#        metric='manhattan')
    # TODO: it might not be a bad idea to inclue a build-method, which would be
    # needed to be run after instantiating the classifier. That would make it a
    # lot faster to train.
    # Alternative: use memory.cache to store the output of the building step!
    classifier = classification.VarmaZissermanClassifier(
        build_classes = [1, 4, 6, 10, 12, 14, 16, 18, 20, 22, 25, 27, 30, 33,
                         35, 41, 45, 48, 50, 59],
        train_classes = [2, 3, 5,  7,  8,  9, 15, 17, 19, 21, 24, 36, 37, 39,
                         43, 44, 47, 52, 54, 58])


#    imgs = memory.cache(dataset.imgs)()
#    feats = feat_ex.transform(imgs)
#    entropies = []
#    for i in range(feats.shape[0]):
#        entropies.append(features.entropy(feats[i], 1))
#    print(np.min(feats), np.max(feats), np.mean(feats))
#    print(np.min(entropies), np.max(entropies), np.mean(entropies))

    # Select classifier
    cross_validate(dataset, feat_ex, classifier)


def bhattacharyya(x, y):
    return (1 - np.dot(np.sqrt(x), np.sqrt(y.T)))**2

if __name__ == '__main__':
    run()
