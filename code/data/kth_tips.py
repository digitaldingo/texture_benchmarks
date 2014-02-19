import os
import numpy as np
from scipy.misc import imread
from sklearn.cross_validation import StratifiedShuffleSplit

from .base import BaseDataset


_URLS = [
    'http://www.nada.kth.se/cvap/databases/kth-tips/kth_tips_grey_200x200.tar',
]
_SHA1S = [
    '6ea7564ef80bc886c1f75a7daaf3f3062d19cafb',
]
_CLASS_NAMES = [
    'aluminium_foil', 'brown_bread', 'corduroy', 'cotton', 'cracker', 'linen',
    'orange_peel', 'sandpaper', 'sponge', 'styrofoam',
]
_IMG_PREFIX_NOS = [15, 48, 42, 46, 60, 44, 55, 6, 21, 20]


class KTH_TIPS(BaseDataset):
    """
    Grayscale version of KTH-TIPS [1].
    http://www.nada.kth.se/cvap/databases/kth-tips

    Reference:
    [1]: Hayman, E.; Caputo, B.; Fritz, M.; Eklundh, J-O., "On the Significance
         of Real-World Conditions for Material Classification", European
         Conference on Computer Vision 2004, Lecture Notes in Computer Science
         Volume 3024, 2004, pp 253-266
    """

    name = 'kth_tips'
    class_names = _CLASS_NAMES
    n_classes = 10
    n_imgs_per_class = 81
    n_imgs = n_classes * n_imgs_per_class

    def __init__(self):
        self.data_dir = os.path.join(self._data_root, self.name)
        self._download_data(_URLS, self.data_dir, sha1s=_SHA1S)
        self.labels = np.hstack([np.ones(self.n_imgs_per_class) * i
                                 for i in range(self.n_classes)])

    def img(self, i):
        class_no = i / self.n_imgs_per_class
        prefix_no = _IMG_PREFIX_NOS[class_no]
        scale_no = (i % self.n_imgs_per_class) / 9 + 1
        img_no = (i % self.n_imgs_per_class) % 9 + 1
        folder = os.path.join('KTH_TIPS', self.class_names[class_no])
        filename = '%.2d-scale_%i_im_%i_grey.png' % (prefix_no, scale_no,
                                                     img_no)
        img = imread(os.path.join(self.data_dir, folder, filename))
        img = img.astype(float)/255
        return img

    def imgs(self):
        return np.array(map(self.img, range(self.n_imgs)))

    def splits(self, n_train_class_imgs=40, n_splits=5, random_state=0):
        train_size = float(n_train_class_imgs)/self.n_imgs_per_class
        test_size = 1-train_size
        s = StratifiedShuffleSplit(self.labels, n_iter=n_splits,
                                   train_size=train_size, test_size=test_size,
                                   random_state=random_state)
        return s
