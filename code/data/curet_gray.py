import os
import itertools
import numpy as np
from scipy.misc import imread
from sklearn.cross_validation import StratifiedShuffleSplit

from .base import BaseDataset


_URLS = [
    'http://www.robots.ox.ac.uk/~vgg/research/texclass/data/curetgrey.zip',
]
_SHA1S = [
    '36efb9be053917b88186e2ce1f07c0e8bf0e36eb',
]
_CLASS_NAMES = [
    'Felt', 'Polyester', 'Terrycloth', 'Rough plastic', 'Leather', 'Sandpaper',
    'Velvet', 'Pebbles', 'Frosted glass', 'Plaster_a', 'Plaster_b',
    'Rough paper', '13-Artificial grass', 'Roof shingle', 'Aluminum foil',
    'Cork', 'Rough tile', 'Rug_a', 'Rug_b', 'Styrofoam', 'Sponge', 'Lambswool',
    'Lettuce leaf', 'Rabbit fur', 'Quarry tile', 'Loofa', 'Insulation',
    'Crumpled paper', 'Polyester (zoomed)', 'Plaster_b (zoomed)',
    'Rough paper (zoomed)', 'Roof shingle (zoomed)', 'Slate_a', 'Slate_b',
    'Painted spheres', 'Limestone', 'Brick_a', 'Ribbed paper', 'Human skin',
    'Straw', 'Brick_b', 'Corduroy', 'Salt crystals', 'Linen', 'Concrete_a',
    'Cotton', 'Stones', 'Brown bread', 'Concrete_b', 'Concrete_c', 'Corn husk',
    'White bread', 'Soleirolia plant', 'Wood_a', 'Orange peel', 'Wood_b',
    'Peacock feather', 'Tree bark', 'Cracker_a', 'Cracker_b', 'Moss',
]
IMG_NOS = [
    2, 11, 14, 16, 17, 19, 22, 25, 26, 28, 30, 32, 33, 35, 36, 37, 39, 41, 42,
    43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 57, 59, 61, 63, 65, 68,
    71, 73, 76, 78, 79, 81, 84, 85, 87, 90, 93, 94, 96, 97, 99, 101, 102, 103,
    104, 106, 109, 112, 114, 116, 117, 119, 121, 122, 123, 125, 128, 129, 131,
    134, 136, 141, 143, 145, 147, 149, 150, 152, 154, 155, 156, 158, 160, 161,
    165, 175, 178, 179, 181, 183,
]


#def grouper(iterable, n, fillvalue=None):
#    "Collect data into fixed-length chunks or blocks"
#    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
#    args = [iter(iterable)] * n
#    return itertools.izip_longest(fillvalue=fillvalue, *args)

def grouper(n, iterable):
    it = iter(iterable)
    while True:
       chunk = tuple(itertools.islice(it, n))
       if not chunk:
           return
       yield chunk

def chunks(l, n):
    """
    Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

class CUReTGray(BaseDataset):
    """
    Cropped grayscale version of the CUReT dataset [1].
    http://www.robots.ox.ac.uk/~vgg/research/texclass/

    The original CUReT dataset is from [2].

    Reference:
    [1]: Varma, M.; Zisserman, A., "A Statistical Approach to Material
         Classification Using Image Patch Exemplars," Pattern Analysis and
         Machine Intelligence, IEEE Transactions on , vol.31, no.11,
         pp.2032,2047, Nov. 2009
    [2]: Dana, K.J.; Nayar, S.K.; Van Ginneken, B.; Koenderink, J.J.,
         "Reflectance and texture of real-world surfaces," Computer Vision and
         Pattern Recognition, 1997. Proceedings., 1997 IEEE Computer Society
         Conference on , vol., no., pp.151,157, 17-19 Jun 1997
    """

    name = 'curet_gray'
    class_names = _CLASS_NAMES
    n_classes = 61
    n_imgs_per_class = 92
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
        img_idx = i % self.n_imgs_per_class
        folder = os.path.join('curetgrey', 'sample%.2d' % (class_no+1))
        filename = '%.2d-%.3d.png' % (class_no+1, IMG_NOS[img_idx])
        img = imread(os.path.join(self.data_dir, folder, filename))
        img = img.astype(float)/255
        return img

    def imgs(self):
        #return np.array([self.img(i) for i in range(self.n_imgs)])
        return np.array(map(self.img, range(self.n_imgs)))

    def imgs_in_chunks(self, chunk_size=100):
        imgs = chunks(map(self.img, range(self.n_imgs)), chunk_size)
        #return grouper(chunk_size, map(self.img, range(self.n_imgs)))
        return list(imgs)

    def splits(self, n_train_class_imgs=43, n_splits=5, random_state=0):
        train_size = float(n_train_class_imgs)/self.n_imgs_per_class
        test_size = 1-train_size
        s = StratifiedShuffleSplit(self.labels, n_iter=n_splits,
                                   train_size=train_size, test_size=test_size,
                                   random_state=random_state)
        return s
