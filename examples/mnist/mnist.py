import gzip
import numpy
import os

from mitax.dataflow.base import RNGDataFlow
import tqdm
from six.moves import urllib

def download(url, dir, filename=None, expect_size=None):
    """
    Download URL to a directory.
    Will figure out the filename automatically from URL, if not given.
    """
    os.makedirs(dir, exist_ok=True)

    if filename is None:
        filename = url.split('/')[-1]
    fpath = os.path.join(dir, filename)

    if os.path.isfile(fpath):
        if expect_size is not None and os.stat(fpath).st_size == expect_size:
            print("File {} exists! Skip download.".format(filename))
            return fpath
        else:
            print("File {} exists. Will overwrite with a new download!".format(filename))

    def hook(t):
        last_b = [0]

        def inner(b, bsize, tsize=None):
            if tsize is not None:
                t.total = tsize
            t.update((b - last_b[0]) * bsize)
            last_b[0] = b
        return inner
    try:
        with tqdm.tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
            fpath, _ = urllib.request.urlretrieve(url, fpath, reporthook=hook(t))
        statinfo = os.stat(fpath)
        size = statinfo.st_size
    except IOError:
        print("Failed to download {}".format(url))
        raise
    assert size > 0, "Downloaded an empty file from {}!".format(url)

    if expect_size is not None and size != expect_size:
        print("File downloaded from {} does not match the expected size!".format(url))
        print("You may have downloaded a broken file, or the upstream may have modified the file.")

    # TODO human-readable size
    print('Succesfully downloaded ' + filename + ". " + str(size) + ' bytes.')
    return fpath

def maybe_download(url, work_directory):
    """Download the data from Yann's website, unless it's already here."""
    filename = url.split('/')[-1]
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        print("Downloading to {}...".format(filepath))
        download(url, work_directory)
    return filepath


def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(filename):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                'Invalid magic number %d in MNIST image file: %s' %
                (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        data = data.astype('float32') / 255.0
        return data


def extract_labels(filename):
    """Extract the labels into a 1D uint8 numpy array [index]."""
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                'Invalid magic number %d in MNIST label file: %s' %
                (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8)
        return labels


class Mnist(RNGDataFlow):
    """
    Produces [image, label] in MNIST dataset,
    image is 28x28 in the range [0,1], label is an int.
    """

    _DIR_NAME = 'mnist_data'
    _SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'

    def __init__(self, train_or_test, shuffle=True, dir=None):
        """
        Args:
            train_or_test (str): either 'train' or 'test'
            shuffle (bool): shuffle the dataset
        """
        if dir is None:
            dir = os.path.realpath(self._DIR_NAME)
        else:
            dir = os.path.join(dir, self._DIR_NAME)
        assert train_or_test in ['train', 'test']
        self.train_or_test = train_or_test
        self.shuffle = shuffle

        def get_images_and_labels(image_file, label_file):
            f = maybe_download(self._SOURCE_URL + image_file, dir)
            images = extract_images(f)
            f = maybe_download(self._SOURCE_URL + label_file, dir)
            labels = extract_labels(f)
            assert images.shape[0] == labels.shape[0]
            return images, labels

        if self.train_or_test == 'train':
            self.images, self.labels = get_images_and_labels(
                'train-images-idx3-ubyte.gz',
                'train-labels-idx1-ubyte.gz')
        else:
            self.images, self.labels = get_images_and_labels(
                't10k-images-idx3-ubyte.gz',
                't10k-labels-idx1-ubyte.gz')

    def __len__(self):
        return self.images.shape[0]

    def __iter__(self):
        idxs = list(range(self.__len__()))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            img = self.images[k].reshape((28, 28))
            label = self.labels[k]
            yield [img, label]