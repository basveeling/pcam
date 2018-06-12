"""PatchCamelyon(PCam) dataset
Small 96x96 patches from histopathology slides from the Camelyon16 dataset.

Please consider citing [1] when used in your publication:
- [1] B. S. Veeling, J. Linmans, J. Winkens, T. Cohen, M. Welling. "Rotation Equivariant CNNs for Digital Pathology". arXiv [cs.CV] (2018), (available at http://arxiv.org/abs/1806.03962).


Author: Bastiaan Veeling
Source: https://github.com/basveeling/pcam
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import pandas as pd
from keras.utils import HDF5Matrix
from keras.utils.data_utils import get_file
from keras import backend as K


def get_unzip_file(fname,
             origin,
             untar=False,
             md5_hash=None,
             file_hash=None,
             cache_subdir='datasets',
             hash_algorithm='auto',
             extract=False,
             archive_format='auto',
             cache_dir=None):
    import gzip
    import shutil
    get_file()
    with open('file.txt', 'rb') as f_in, gzip.open('file.txt.gz', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

def load_data():
    """Loads PCam dataset.

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    dirname = os.path.join('datasets', 'pcam')
    base = 'https://drive.google.com/uc?export=download&id='
    try:
        y_train = HDF5Matrix(get_file('camelyonpatch_level_2_split_train_y.h5', origin= base+ '1269yhu3pZDP8UYFQs-NYs3FPwuK-nGSG', cache_subdir=dirname, archive_format='gzip'), 'y')
        x_valid = HDF5Matrix(get_file('camelyonpatch_level_2_split_valid_x.h5', origin= base+ '1hgshYGWK8V-eGRy8LToWJJgDU_rXWVJ3', cache_subdir=dirname, archive_format='gzip'), 'x')
        y_valid = HDF5Matrix(get_file('camelyonpatch_level_2_split_valid_y.h5', origin= base+ '1bH8ZRbhSVAhScTS0p9-ZzGnX91cHT3uO', cache_subdir=dirname, archive_format='gzip'), 'y')
        x_test = HDF5Matrix(get_file('camelyonpatch_level_2_split_test_x.h5', origin= base+ '1qV65ZqZvWzuIVthK8eVDhIwrbnsJdbg_', cache_subdir=dirname, archive_format='gzip'), 'x')
        y_test = HDF5Matrix(get_file('camelyonpatch_level_2_split_test_y.h5', origin= base+ '17BHrSrwWKjYsOgTMmoqrIjDy6Fa2o_gP', cache_subdir=dirname, archive_format='gzip'), 'y')

        meta_train = pd.read_csv(get_file('camelyonpatch_level_2_split_train_meta.csv', origin= base+ '1XoaGG3ek26YLFvGzmkKeOz54INW0fruR', cache_subdir=dirname))
        meta_valid = pd.read_csv(get_file('camelyonpatch_level_2_split_valid_meta.csv', origin= base+ '16hJfGFCZEcvR3lr38v3XCaD5iH1Bnclg', cache_subdir=dirname))
        meta_test = pd.read_csv(get_file('camelyonpatch_level_2_split_test_meta.csv', origin= base+ '19tj7fBlQQrd4DapCjhZrom_fA4QlHqN4', cache_subdir=dirname))
        x_train = HDF5Matrix(get_file('camelyonpatch_level_2_split_train_x.h5', origin= base+ '1Ka0XfEMiwgCYPdTI-vv6eUElOBnKFKQ2', cache_subdir=dirname, archive_format='gzip'), 'x')
    except OSError:
        raise NotImplementedError('Direct download currently not working. Please go to https://drive.google.com/drive/folders/1gHou49cA1s5vua2V5L98Lt8TiWA3FrKB and press download all. Then place files (ungzipped) in ~/.keras/datasets/pcam.')
        
    if K.image_data_format() == 'channels_first':
        raise NotImplementedError()

    return (x_train, y_train, meta_train), (x_valid, y_valid, meta_valid), (x_test, y_test, meta_test)


if __name__ == '__main__':
    (x_train, y_train, meta_train), (x_valid, y_valid, meta_valid), (x_test, y_test, meta_test) = load_data()
    print(len(x_train))
