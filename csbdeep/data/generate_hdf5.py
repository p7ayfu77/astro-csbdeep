# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, absolute_import, division
from numpy import random
from six.moves import range, zip, map, reduce, filter
from six import string_types

import numpy as np
import sys, os, warnings

from tqdm import tqdm
from ..utils import _raise, consume, compose, normalize_mi_ma, axes_dict, axes_check_and_normalize, choice
from ..utils.six import Path
from ..io import save_training_data

from .transform import Transform, permute_axes, broadcast_target
from .generate import _memory_check, no_background_patches, norm_percentiles, sample_patches_from_multiple_stacks
import tempfile
import h5py

def create_patches_hdf5(
        raw_data,
        patch_size,
        n_patches_per_image,
        patch_axes    = None,
        save_file     = None,
        transforms    = None,
        patch_filter  = no_background_patches(),
        normalization = norm_percentiles(),
        overlap       = True,
        verbose       = True,
        shuffle       = True,
        collapse_channel = False,
        chunk_samples = 256
    ):
    """Create normalized training data saved to HDF5 file on disk, to be used for neural network training.

    Parameters
    ----------
    raw_data : :class:`RawData`
        Object that yields matching pairs of raw images.
    patch_size : tuple
        Shape of the patches to be extraced from raw images.
        Must be compatible with the number of dimensions and axes of the raw images.
        As a general rule, use a power of two along all XYZT axes, or at least divisible by 8.
    n_patches_per_image : int
        Number of patches to be sampled/extracted from each raw image pair (after transformations, see below).
    patch_axes : str or None
        Axes of the extracted patches. If ``None``, will assume to be equal to that of transformed raw data.
    save_file : str or None
        File name to save training data to disk in ``.npz`` format (see :func:`csbdeep.io.save_training_data`).
        If ``None``, data will not be saved.
    transforms : list or tuple, optional
        List of :class:`Transform` objects that apply additional transformations to the raw images.
        This can be used to augment the set of raw images (e.g., by including rotations).
        Set to ``None`` to disable. Default: ``None``.
    patch_filter : function, optional
        Function to determine for each image pair which patches are eligible to be extracted
        (default: :func:`no_background_patches`). Set to ``None`` to disable.
    normalization : function, optional
        Function that takes arguments `(patches_x, patches_y, x, y, mask, channel)`, whose purpose is to
        normalize the patches (`patches_x`, `patches_y`) extracted from the associated raw images
        (`x`, `y`, with `mask`; see :class:`RawData`). Default: :func:`norm_percentiles`.
    overlap: bool, optional
        Flag indicating if patches are will overlap when randomly samples. If set False random selection is disabled
        and `n_patches_per_image` must be equal to or less than number of pacthes expected for patch_size across all images.
    shuffle : bool, optional
        Randomly shuffle all extracted patches.
    collapse_channel : bool, optional
        Collapse Channel dimension, useful when wanting to process all data from RGB data as one channel.
    chunk_samples : int, optional
        Number of samples per HDF5 chunk `(chunk_samples,)+tuple(patch_size))`. This can be aligned with the expected read batch size when training for better disk & memory IO performance
    verbose : bool, optional
        Display overview of images, transforms, etc.

    Raises
    ------
    ValueError
        Various reasons.

    Example
    -------
    >>> raw_data = RawData.from_folder(basepath='data', source_dirs=['source1','source2'], target_dir='GT', axes='ZYX')
    >>> create_patches_hdf5(raw_data, patch_size=(32,128,128), n_patches_per_image=16)

    Todo
    ----
    - Test if perf is better with chunks of n_patches_per_image for HDF5

    """
    ## images and transforms
    if transforms is None:
        transforms = []
    transforms = list(transforms)
    if patch_axes is not None:
        transforms.append(permute_axes(patch_axes))
    if len(transforms) == 0:
        transforms.append(Transform.identity())

    if normalization is None:
        normalization = lambda patches_x, patches_y, x, y, mask, channel: (patches_x, patches_y)

    image_pairs, n_raw_images = raw_data.generator(), raw_data.size
    tf = Transform(*zip(*transforms)) # convert list of Transforms into Transform of lists
    image_pairs = compose(*tf.generator)(image_pairs) # combine all transformations with raw images as input

    collapse_multiplier = 1
    out_patch_size = patch_size
    if collapse_channel:
        temp_image_pairs = compose(*tf.generator)(raw_data.generator())
        _x,_y,_axes,_mask = next(temp_image_pairs) # get the first entry from the generator
        _axes = axes_check_and_normalize(_axes,len(patch_size))
        _channel = axes_dict(_axes)['C']
        collapse_multiplier = _x.shape[_channel]
        out_patch_size = list(out_patch_size)
        out_patch_size[_channel] = 1
        out_patch_size = tuple(out_patch_size)
        del _x,_y,_axes,_mask

    n_transforms = np.prod(tf.size)
    n_images = n_raw_images * n_transforms    
    n_patches = n_images * n_patches_per_image

    ## summary
    if verbose:
        print('='*66)
        print('%5d raw images x %4d transformations   = %5d images' % (n_raw_images,n_transforms,n_images))
        print('%5d images     x %4d patches per image = %5d patches in total' % (n_images,n_patches_per_image,n_patches))
        if collapse_channel:
            print('%5d patches    x %4d channel per patch = %5d collased patches in total' % (n_patches,collapse_multiplier,n_patches*collapse_multiplier))
        print('='*66)
        print('Input data:')
        print(raw_data.description)
        print('='*66)
        print('Transformations:')
        for t in transforms:
            print('{t.size} x {t.name}'.format(t=t))
        print('='*66)
        print('Patch size:')
        print(" x ".join(str(p) for p in patch_size))
        if collapse_channel:
            print('Collapsed Patch size:')
            print(" x ".join(str(p) for p in out_patch_size))
        print('=' * 66)

    sys.stdout.flush()

    datashape = (n_patches*collapse_multiplier,)+tuple(out_patch_size)
    #TODO: Test if perf is better with chunks of n_patches_per_image
    #chunks=((n_patches_per_image*collapse_multiplier,)+tuple(out_patch_size))
    if shuffle:
        chunks=((1,)+tuple(out_patch_size))
    else:
        chunks=((chunk_samples,)+tuple(out_patch_size))
    
    parent_path = Path(save_file).parent
    with tempfile.TemporaryFile(dir=parent_path) as temporaryFile:
        temporaryFileName = temporaryFile.name
    
    with h5py.File(temporaryFileName, 'w', libver='latest') as f:
        
        X = f.create_dataset("X", datashape, maxshape=datashape, chunks=chunks, dtype='float32')
        Y = f.create_dataset("Y", datashape, maxshape=datashape, chunks=chunks, dtype='float32')        

        ## sample patches from each pair of transformed raw images
        for i, (x,y,_axes,mask) in tqdm(enumerate(image_pairs),total=n_images,disable=(not verbose)):
            if i >= n_images:
                warnings.warn('more raw images (or transformations thereof) than expected, skipping excess images.')
                break
            if i==0:
                axes = axes_check_and_normalize(_axes,len(patch_size))
                channel = axes_dict(axes)['C']
            # checks
            # len(axes) >= x.ndim or _raise(ValueError())
            axes == axes_check_and_normalize(_axes) or _raise(ValueError('not all images have the same axes.'))
            x.shape == y.shape or _raise(ValueError())
            mask is None or mask.shape == x.shape or _raise(ValueError())
            (channel is None or (isinstance(channel,int) and 0<=channel<x.ndim)) or _raise(ValueError())
            channel is None or patch_size[channel]==x.shape[channel] or _raise(ValueError('extracted patches must contain all channels.'))

            _Y,_X = sample_patches_from_multiple_stacks((y,x), patch_size, n_patches_per_image, mask, patch_filter, overlap=overlap)
            
            if not overlap and len(_X) != n_patches_per_image:
                raise ValueError("Missmatched number of patches per image when no overlap")

            _Xn, _Yn = normalization(_X,_Y, x,y,mask,channel)
            
            if collapse_channel:
                s = slice(i*n_patches_per_image*collapse_multiplier,(i+1)*n_patches_per_image*collapse_multiplier)
                X[s] = np.moveaxis(np.concatenate(_Xn, axis=channel),channel,0)[...,np.newaxis]
                Y[s] = np.moveaxis(np.concatenate(_Yn, axis=channel),channel,0)[...,np.newaxis]
            else:
                s = slice(i*n_patches_per_image,(i+1)*n_patches_per_image)
                X[s], Y[s] = _Xn, _Yn

        axes = 'S'+axes
        X.attrs['axes'] = axes
        Y.attrs['axes'] = axes

    
    if shuffle:
        print('Shuffling HDF5 data via copy. This can take time for large datasets...')
        out_chunks=((chunk_samples,)+tuple(out_patch_size))
        hdf5_copy_shuffle(temporaryFileName,save_file,datashape=datashape,chunks=out_chunks)
        os.remove(temporaryFileName)
    else:
        os.rename(temporaryFileName,save_file)

    print('Saved as hdf5 data to %s.' % str(Path(save_file)))

# Misc

def hdf5_copy_shuffle(tempfile,outfile,datashape=None,chunks=None):
    with h5py.File(tempfile, 'r') as hdf5file_temp:
        X_dset = hdf5file_temp['X']
        Y_dset = hdf5file_temp['Y']

        len(X_dset) == len(Y_dset) or _raise(ValueError("X and Y must have same length"))

        if datashape is None:
            datashape = X_dset.shape
        print("Destination Shape", datashape)
        if chunks is None:
            chunks = X_dset.chunks
        print("Destination Chunks", chunks)

        n_samples = len(X_dset)
        shuffle_indx = np.arange(n_samples)
        random.shuffle(shuffle_indx)
        
        with h5py.File(outfile, 'w') as hdf5file_out:
            X = hdf5file_out.create_dataset("X", datashape, maxshape=datashape, chunks=chunks, dtype='float32')
            Y = hdf5file_out.create_dataset("Y", datashape, maxshape=datashape, chunks=chunks, dtype='float32')
                
            for dest_idx, source_idx in tqdm(enumerate(shuffle_indx),total=n_samples,disable=False):
                X[dest_idx,...] = X_dset[source_idx,...]
                Y[dest_idx,...] = Y_dset[source_idx,...]
            
            X.attrs['axes'] = X_dset.attrs['axes']
            Y.attrs['axes'] = Y_dset.attrs['axes']


# Data Wrapper

class HDF5Data():

    @staticmethod
    def from_hdf5(file, validation_split=0, channels=slice(None,), shuffled_read=True):
        assert 0 <= validation_split < 1        
        hdf5file = h5py.File(file, 'r')
        X_dset = hdf5file['X']
        Y_dset = hdf5file['Y']
        len(X_dset) == len(Y_dset) or _raise(ValueError("X and Y must have same length"))
        n_samples = len(X_dset)
        hdf5file.close()

        shuffle_indx = np.arange(n_samples)
        if shuffled_read:           
            random.shuffle(shuffle_indx)

        if validation_split > 0:
            n_val   = int(round(n_samples * validation_split))
            n_train = n_samples - n_val
            assert 0 < n_val and 0 < n_train
            val_idx = shuffle_indx[-n_val:]
            train_idx = shuffle_indx[:n_train]

        return HDF5Data(file, sample_indx=train_idx, channels=channels), HDF5Data(file, sample_indx=val_idx, channels=channels)


    def __init__(self, file, sample_indx=None,channels=slice(None,)):
        
        hdf5file = h5py.File(file, 'r')
        X_dset = hdf5file['X']
        Y_dset = hdf5file['Y']

        if sample_indx is None:
            sample_indx = list(range(len(X_dset)))

        self.hdf5file = hdf5file
        self.channels_slice = channels
        self.X, self.Y = X_dset, Y_dset
        self.sample_indx = sample_indx
    
    def __getitem__(self, args):

        idx = np.atleast_1d(self.sample_indx[args])

        sort_index = np.argsort(idx)
        rev_index = np.argsort(sort_index)

        X_t = self.X[idx[sort_index]]
        Y_t = self.Y[idx[sort_index]]
        return X_t[rev_index][...,self.channels_slice], Y_t[rev_index][...,self.channels_slice]

    def set_channel(self, channels=slice(None,)):
        self.channels_slice = channels

    def __del__(self):
        self.hdf5file.close()

    @property
    def axes(self):
        return self.X.attrs['axes'], self.Y.attrs['axes']

    @property
    def shape(self):
        return (len(self.sample_indx),) + self.X.shape[1:]
    
    @property
    def ndim(self):
        return self.X.ndim

    @property
    def dtype(self):
        return self.X.dtype

    def __len__(self):
        return len(self.sample_indx)