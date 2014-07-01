#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sklearn
from skimage import exposure

import data
import ipcv
from ipcv import josi_hist
import features

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
    plt.savefig(path, dpi=120, bbox_inches='tight')


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
    fig, axarr = plt.subplots(3, max_n_imgs, figsize=(45, 11))
    plt.suptitle('Difficult images in %s using %s' % (dataset.name,
                                                      feat_ex.name),
                 fontsize=12)
    for n, (idx, nearest_idx) in enumerate(difficult_idxs[:max_n_imgs]):
        img = dataset.img(idx)
        img_save(img, os.path.join(imgdir, '%.2d.png' % n))

        axarr[0, n].imshow(img, cmap='gray')
        axarr[0, n].axis('off')
        axarr[0, n].set_title('Test image: ' + dataset.class_names[dataset.label(idx)])

        plot_entropy(axarr[1, n], img)

        img_incorrect = dataset.img(nearest_idx)
        img_save(img_incorrect, os.path.join(imgdir,
                                             '%.2d_nearest_incorrect.png' % n))
        axarr[2, n].imshow(img_incorrect, cmap='gray')
        axarr[2, n].axis('off')
        axarr[2, n].set_title('Wrong match: ' + dataset.class_names[dataset.label(nearest_idx)])

    fig_save(os.path.join(outdir, '%s_%s.pdf' % (dataset.name, feat_ex.name)))

def plot_entropy(ax, img):
    scales = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 24]
    tonal_scale = 0.1
    alphas = [1.0]
    n_bins = 32
    def intensity_hist(img, scale):
        return features.intensity_hist(img, scale, n_bins, tonal_scale)
    def gradient_hist(img, scale):
        return features.gradient_hist(img, scale, n_bins, tonal_scale)
    def shape_index_hist(img, scale):
        return features.shape_index_hist(img, scale, n_bins, tonal_scale)
    entropy_data = entropy_diff(img, scales, alphas, intensity_hist)
    for alpha_idx, alpha in enumerate(alphas):
        ax.plot(scales, entropy_data[:,alpha_idx], linestyle='-', label=r'${\bf p}_I , \alpha=' + str(alpha) +r'$')
    entropy_data = entropy_diff(img, scales, alphas, gradient_hist)
    for alpha_idx, alpha in enumerate(alphas):
        ax.plot(scales, entropy_data[:,alpha_idx], linestyle='--', label=r'${\bf p}_\theta , \alpha=' + str(alpha) +r'$')
    entropy_data = entropy_diff(img, scales, alphas, shape_index_hist)
    for alpha_idx, alpha in enumerate(alphas):
        ax.plot(scales, entropy_data[:,alpha_idx], linestyle=':', label=r'${\bf p}_s , \alpha=' + str(alpha) +r'$')
    ax.legend(prop={'size':8})
    ax.set_ylim([0, 1.2])
    ax.set_xlim([np.min(scales), np.max(scales)])
    ax.set_title(r'$\frac{\partial S_\alpha({\bf p}(\sigma))}{\partial \sigma}'
                 + r',\beta='+ str(tonal_scale)
                 + r',N=' + str(n_bins)
                 + r'$')


def entropy_diff(img, scales, alphas, hist_extractor, diff_step=0.05):
    entropy_diffs = np.empty((len(scales), len(alphas)))
    for scale_idx, scale in enumerate(scales):
        feat1 = hist_extractor(img, scale-scale*diff_step)
        feat2 = hist_extractor(img, scale+scale*diff_step)
        for alpha_idx, alpha in enumerate(alphas):
            entropy1 = features.entropy(feat1, alpha)
            entropy2 = features.entropy(feat2, alpha)
            entropy_diffs[scale_idx, alpha_idx] = (entropy1-entropy2)/(2*diff_step)
    return entropy_diffs


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



def multiscale_shape_index():
    dataset = data.UIUCTex()
    imgs = memory.cache(dataset.imgs)()
    img_idx = 401
    img = imgs[img_idx]
    dirpath = os.path.join('multiscale_shape_index', dataset.name, 'img%.3d'%img_idx)
    img_save(img, os.path.join(dirpath, 'img.png'))
    scales = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38]
    for scale in scales:
        si, si_c, si_o, si_om = ipcv.shape_index(img, scale, orientations=True)
        img_save(si, os.path.join(dirpath, 'si_scale%.2d.png' % scale))
        img_save(si_c, os.path.join(dirpath, 'si-c_scale%.2d.png' % scale))
        img_save(si_o, os.path.join(dirpath, 'si-o_scale%.2d.png' % scale))
        img_save(si_om, os.path.join(dirpath, 'si-om_scale%.2d.png' % scale))


def oriented_shape_index():
    dataset = data.UIUCTex()
    imgs = memory.cache(dataset.imgs)()
    scales = 1*2.0**np.arange(5)
    feat_ex = features.OrientedShapeIndexHistograms(**{
        'scales': scales,
        'n_bins': 24, #50, #24,
        'tonal_scale': .2,
        'ori_n_bins': 16, #50, #8,
        'ori_tonal_scale': 0.25,
        'ori_detect_scale': 4,
        'norm': 'l1',
        'joint_hist': False,
        'flat': False,
    })

    img_idxs = [401]
    for img_idx in img_idxs:
        img = imgs[img_idx]
        print(img_idx)
        dirpath = os.path.join('oriented_shape_index', dataset.name, 'img%.3d'%img_idx)
        img_save(img, os.path.join(dirpath, 'img.png'))
        hists = feat_ex.transform(img)
#        print(hists.shape)
        for idx, scale in zip(range(hists.shape[2]), scales):
#            print(hists[..., i].shape)
            img_save(hists[..., idx], os.path.join(dirpath, 'josi_scale%i.png'%scale))



def multiscale_entropy(img, feat, scales, tonal_scale, n_bins, alpha):
    def entropy(img, scales, alpha, hist_extractor):
        entropies = np.empty((len(scales),))
        for scale_idx, scale in enumerate(scales):
            feat = hist_extractor(img, scale)
            entropy = features.entropy(feat, alpha)
            entropies[scale_idx] = entropy
        return entropies

    if feat == 'si':
        def gradient_hist(img, scale):
            return features.gradient_hist(img, scale, n_bins, tonal_scale)
        entropies = entropy(img, scales, alpha, gradient_hist)
    elif feat == 'go':
        def shape_index_hist(img, scale):
            return features.shape_index_hist(img, scale, n_bins, tonal_scale)
        entropies = entropy(img, scales, alpha, shape_index_hist)
    else:
        raise NotImplementedError
    return entropies


def hdr_entropy_plots():
    dataset = data.DIKUScaleSeq()

    feat = 'go'
    tonal_scale = 1.5
    alpha = 1.0
    n_bins = 16

    for class_idx in range(dataset.n_classes):

        img = dataset.img(dataset.n_imgs_per_class * class_idx)
        base_img_width = img.shape[1]
        fig = plt.figure(1)
        fig.clf()
        ax = fig.gca()
        ax.set_xscale('log')
        feat_name = 's' if feat == 'si' else r'\theta'
        ax.set_title(r'$S({\bf p}_%s(\sigma))' % feat_name
                     + r',\beta='+ str(tonal_scale)
                     + r',N=' + str(n_bins)
                     + r'$')

        for idx in range(dataset.n_imgs_per_class):
            img_no = idx+1
            print(idx)
            img = dataset.img(dataset.n_imgs_per_class * class_idx + idx)

            # exposure normalization
            p2, p98 = np.percentile(img, (2, 98))
            img = exposure.rescale_intensity(img, in_range=(p2, p98))

            # determine scales for entropy measures
            img_width = float(img.shape[1])
            scale_factor = base_img_width / img_width
            scale_offset = -scale_factor
            scales = np.logspace(.001, np.log2(img_width/10.), num=25, base=2.)
            print(scales)
            # calculate entropies
            entropies = multiscale_entropy(img, feat, scales, tonal_scale,
                                           n_bins, alpha)
            # plot
            scales_offset = scales*scale_factor+scale_offset
            label = r'%i' % img_no
            ax.plot(scales_offset, entropies, linestyle='-', label=label)
            ax.legend(prop={'size':9})

            # output images
            plot_dir = os.path.join('entropy_plots', dataset.name,
                                    dataset.class_names[class_idx]+'-'+feat)
            img_save(img, os.path.join(plot_dir,
                                       'img_%.3d.png' % img_no))
            fig_save(os.path.join(plot_dir,
                                  'entropy_%.3d.png' % img_no))



if __name__ == '__main__':
#    oriented_shape_index()
#    multiscale_shape_index()
#    entropy()
    hdr_entropy_plots()
#    josih()
