#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sklearn

import data
from ipcv import josi_hist

from joblib import Memory
memory = Memory(cachedir=os.environ['CACHE_ROOT'],
                verbose=int(os.environ['VERBOSITY']))


def img_save(img, path):
    path = os.path.join(os.environ['RESULTS_ROOT'], path)
    dirpath = os.path.dirname(path)
    if len(dirpath) > 0 and not os.path.exists(dirpath):
        os.makedirs(dirpath)
    sp.misc.imsave(path, img)


def fig_save(path):
    path = os.path.join(os.environ['RESULTS_ROOT'], path)
    dirpath = os.path.dirname(path)
    if len(dirpath) > 0 and not os.path.exists(dirpath):
        os.makedirs(dirpath)
    plt.savefig(path, bbox_inches='tight')


def difficult_imgs(dataset, feat_ex, feats, idxs, y_tests, y_preds,
                   max_n_imgs=10):
    outdir = os.path.join('difficult_imgs')
    imgdir = os.path.join(outdir, feat_ex.name, dataset.name)

    # Find images that get the most incorrect classifications
    n_errors = np.zeros((dataset.n_imgs,))
    for test_idx, y_test, y_pred in zip(idxs, y_tests, y_preds):
        n_errors[test_idx[y_pred != y_test]] += 1

    # Find nearest (incorrect) match
    dists = sklearn.metrics.pairwise.pairwise_distances(feats,
                                                        metric='manhattan')
    difficult_idxs = []
    for n, idx in enumerate(np.flatnonzero(n_errors)):
        nearest_idx = np.argsort(dists[idx, :])[1]
        if dataset.label(idx) != dataset.label(nearest_idx):
            # We require difficult images to be incorrect when comparing with
            # all other images. Otherwise, the original incorrect
            # classification may be the result of an unlucky dataset split.
            difficult_idxs.append((idx, nearest_idx))
    difficult_idxs = sorted(difficult_idxs, key=lambda x: n_errors[x[0]])

    # Plot and write to disk
    max_n_imgs = min(max_n_imgs, len(difficult_idxs))
    fig, axarr = plt.subplots(2, max_n_imgs, figsize=(60, 6))
    plt.suptitle('Difficult images in %s using %s' % (dataset.name,
                                                      feat_ex.name),
                 fontsize=12)
    for n, (idx, nearest_idx) in enumerate(difficult_idxs[:max_n_imgs]):
        img = dataset.img(idx)
        img_save(img, os.path.join(imgdir, '%.2d.png' % n))

        axarr[0, n].imshow(img, cmap='gray')
        axarr[0, n].axis('off')
        axarr[0, n].set_title(dataset.class_names[dataset.label(idx)])

        img_incorrect = dataset.img(nearest_idx)
        img_save(img_incorrect, os.path.join(imgdir,
                                             '%.2d_nearest_incorrect.png' % n))
        axarr[1, n].imshow(img_incorrect, cmap='gray')
        axarr[1, n].axis('off')
        axarr[1, n].set_title(dataset.class_names[dataset.label(nearest_idx)])
    fig_save(os.path.join(outdir, '%s_%s.pdf' % (dataset.name, feat_ex.name)))


def josih():
    # Select dataset
    dataset = data.CUReTGray()
#    dataset = data.KTH_TIPS()
#    dataset = data.UIUCTex()

    # Select feature extractor
    scales = 1*2.0**np.arange(4)
    opts_sih = {
        'scales': scales,
        'n_bins': 40,
        'tonal_scale': np.pi/40*3,
        'ori_n_bins': 40,
        'ori_tonal_scale': np.pi/40*3,
        'norm': 'l1',
    }

    imgs = memory.cache(dataset.imgs)()
    for idx, img in enumerate(imgs):
        img_save(img, os.path.join('josih', dataset.name,
                                   '%.3d_img.png' % idx))
        hists = josi_hist(img, **opts_sih)
        print(hists.shape)
        for i in range(hists.shape[2]):
            hist_path = os.path.join('josih', dataset.name,
                                     '%.3d_josih_scale%i.png' % (idx,
                                                                 scales[i]))
            img_save(hists[..., i], hist_path)


if __name__ == '__main__':
    josih()
