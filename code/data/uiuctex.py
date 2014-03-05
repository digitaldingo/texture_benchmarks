import os

import numpy as np
from scipy.misc import imread
from sklearn.cross_validation import StratifiedShuffleSplit

from .base import BaseDataset


_URLS = [
    'http://www-cvr.ai.uiuc.edu/ponce_grp/data/texture_database/T01-T05.zip',
    'http://www-cvr.ai.uiuc.edu/ponce_grp/data/texture_database/T06-T10.zip',
    'http://www-cvr.ai.uiuc.edu/ponce_grp/data/texture_database/T11-T15.zip',
    'http://www-cvr.ai.uiuc.edu/ponce_grp/data/texture_database/T16-T20.zip',
    'http://www-cvr.ai.uiuc.edu/ponce_grp/data/texture_database/T21-T25.zip',
]
_SHA1S = [
    'c0761b76080a9a9200b4a6894e05fabaca1fb280',
    'a83dfd0af0a3edd5d144f7cdd2e6e93eb75c2c4a',
    'aa9879721b40364f7a69e589937afd0848612a0e',
    'fc0d89efe370ee36605336bd112dae8503004dad',
    'f9c2b8afeaa7a5d56e7d135a0a05bd34f818428a',
]

_CLASS_NAMES = [
    'bark1', 'bark2', 'bark3', 'wood1', 'wood2', 'wood3', 'water', 'granite',
    'marble', 'floor1', 'floor2', 'pebbles', 'wall', 'brick1', 'brick2',
    'glass1', 'glass2', 'carpet1', 'carpet2', 'upholstery', 'wallpaper', 'fur',
    'knit', 'corduroy', 'plaid',
]


class UIUCTex(BaseDataset):
    """
    University of Illinois at Urbana-Champaign Texture Database [1]
    http://www-cvr.ai.uiuc.edu/ponce_grp/data/

    Reference:
    [1]: Lazebnik, S.; Schmid, C.; Ponce, J., "A sparse texture representation
         using local affine regions," Pattern Analysis and Machine
         Intelligence, IEEE Transactions on , vol.27, no.8, pp.1265,1278, Aug.
         2005
    """

    name = 'uiuctex'
    class_names = _CLASS_NAMES
    n_classes = 25
    n_imgs_per_class = 40
    n_imgs = n_classes * n_imgs_per_class

    def __init__(self):
        self.data_dir = os.path.join(self._data_root, self.name)
        self._download_data(_URLS, self.data_dir, sha1s=_SHA1S)
        self.labels = np.hstack([np.ones(self.n_imgs_per_class) * i
                                 for i in range(self.n_classes)]).astype(int)

    def label(self, i):
        return self.labels[i]

    def img(self, i):
        class_no = i / self.n_imgs_per_class
        img_no = i % self.n_imgs_per_class
        folder = 'T%.2d_%s' % (class_no+1, self.class_names[class_no])
        filename = 'T%.2d_%.2d.jpg' % (class_no+1, img_no+1)
        img = imread(os.path.join(self.data_dir, folder, filename))
        img = img.astype(float)/255
        return img

    def imgs(self):
        return np.array(map(self.img, range(self.n_imgs)))

    def splits(self, n_train_class_imgs=20, n_splits=100, random_state=0):
        train_size = float(n_train_class_imgs)/self.n_imgs_per_class
        test_size = 1-train_size
        s = StratifiedShuffleSplit(self.labels, n_iter=n_splits,
                                   train_size=train_size, test_size=test_size,
                                   random_state=random_state)
        return s
