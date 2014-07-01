import os
import numpy as np
from scipy.misc import imread
from sklearn.cross_validation import StratifiedShuffleSplit

from .base import BaseDataset


_CLASS_NAMES = [
    'kockums', 'dobelnsgatan', 'dobelnsgatan_small_house', 'shrub_rosjoparken', 'birch_tree_1_university_park', 'birch_tree_trunk_1',
]
_IMG_PREFIX_NOS = [2, 4, 5, 6, 10, 15]


class DIKUScaleSeq(BaseDataset):
    """
    Grayscale version of KTH-TIPS [1].
    http://www.nada.kth.se/cvap/databases/kth-tips

    Reference:
    [1]: Hayman, E.; Caputo, B.; Fritz, M.; Eklundh, J-O., "On the Significance
         of Real-World Conditions for Material Classification", European
         Conference on Computer Vision 2004, Lecture Notes in Computer Science
         Volume 3024, 2004, pp 253-266
    """

    name = 'diku_scale_seq'
    class_names = _CLASS_NAMES
    n_classes = 6
    n_imgs_per_class = 15
    n_imgs = n_classes * n_imgs_per_class

    def __init__(self):
        self.data_dir = os.path.join(self._data_root, self.name)
        self.labels = np.hstack([np.ones(self.n_imgs_per_class) * i
                                 for i in range(self.n_classes)]).astype(int)

    def label(self, i):
        return self.labels[i]

    def img(self, i):
        class_no = i / self.n_imgs_per_class
        img_idx = i % self.n_imgs_per_class+1
        prefix_no = _IMG_PREFIX_NOS[class_no]
        folder = os.path.join('%.2d_%s' % (prefix_no, self.class_names[class_no]), 'sample_a')
        filename = '%.2da_im_reg_15_to_%d.TIF' % (prefix_no, img_idx)
        img = imread(os.path.join(self.data_dir, folder, filename))
        img = img.astype(float)/2**16
        return img

    def imgs(self):
        return np.array(map(self.img, range(self.n_imgs)))

    def splits(self, n_train_class_imgs=40, n_splits=100, random_state=0):
        raise NotImplementedError()
