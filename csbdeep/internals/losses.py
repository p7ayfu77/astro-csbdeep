from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter

from ..utils import _raise, backend_channels_last

import numpy as np
from ..utils.tf import keras_import
K = keras_import('backend')

import tensorflow as tf

def _mean_or_not(mean):
    # return (lambda x: K.mean(x,axis=(-1 if backend_channels_last() else 1))) if mean else (lambda x: x)
    # Keras also only averages over axis=-1, see https://github.com/keras-team/keras/blob/master/keras/losses.py
    return (lambda x: K.mean(x,axis=-1)) if mean else (lambda x: x)


def loss_laplace(mean=True):
    R = _mean_or_not(mean)
    C = np.log(2.0)
    if backend_channels_last():
        def nll(y_true, y_pred):
            n     = K.shape(y_true)[-1]
            mu    = y_pred[...,:n]
            sigma = y_pred[...,n:]
            return R(K.abs((mu-y_true)/sigma) + K.log(sigma) + C)
        return nll
    else:
        def nll(y_true, y_pred):
            n     = K.shape(y_true)[1]
            mu    = y_pred[:,:n,...]
            sigma = y_pred[:,n:,...]
            return R(K.abs((mu-y_true)/sigma) + K.log(sigma) + C)
        return nll


def loss_mae(mean=True):
    R = _mean_or_not(mean)
    if backend_channels_last():
        def mae(y_true, y_pred):
            n = K.shape(y_true)[-1]
            return R(K.abs(y_pred[...,:n] - y_true))
        return mae
    else:
        def mae(y_true, y_pred):
            n = K.shape(y_true)[1]
            return R(K.abs(y_pred[:,:n,...] - y_true))
        return mae


def loss_mse(mean=True):
    R = _mean_or_not(mean)
    if backend_channels_last():
        def mse(y_true, y_pred):
            n = K.shape(y_true)[-1]
            return R(K.square(y_pred[...,:n] - y_true))
        return mse
    else:
        def mse(y_true, y_pred):
            n = K.shape(y_true)[1]
            return R(K.square(y_pred[:,:n,...] - y_true))
        return mse


def loss_thresh_weighted_decay(loss_per_pixel, thresh, w1, w2, alpha):
    def _loss(y_true, y_pred):
        val = loss_per_pixel(y_true, y_pred)
        k1 = alpha * w1 + (1 - alpha)
        k2 = alpha * w2 + (1 - alpha)
        return K.mean(K.tf.where(K.tf.less_equal(y_true, thresh), k1 * val, k2 * val),
                      axis=(-1 if backend_channels_last() else 1))
    return _loss


def loss_hdr(mean=True):
    R = _mean_or_not(mean)
    if backend_channels_last():
        def hdr(y_true, y_pred):
            n = K.shape(y_true)[-1]
            return R(K.abs(K.tanh(y_pred[...,:n])-K.tanh(y_true)))
        return hdr
    else:
        def hdr(y_true, y_pred):
            n = K.shape(y_true)[1]
            return R(K.abs(K.tanh(y_pred[:,:n,...])-K.tanh(y_true)))
        return hdr


def loss_msssim(mean=False):
    power_factors=(0.0448, 0.2856, 0.3001)
    if backend_channels_last():
        def msssim(y_true, y_pred):
            n = K.shape(y_true)[-1]
            return  1 - tf.image.ssim_multiscale(img1=y_true, img2=y_pred[...,:n], max_val=1.0, power_factors=power_factors)
        return msssim
    else:
        def msssim(y_true, y_pred):
            n = K.shape(y_true)[1]
            return 1 - tf.image.ssim_multiscale(img1=y_true, img2=y_pred[:,:n,...], max_val=1.0, power_factors=power_factors)
        return msssim


def loss_msssimhdr(mean=False):
    alpha = 0.125
    compensation=10.0
    power_factors=(0.0448, 0.2856, 0.3001)
    if backend_channels_last():
        def msssimhdr(y_true, y_pred):
            n = K.shape(y_true)[-1]
            axis = tuple((d for d in range(K.ndim(y_true)) if d != 0))
            ssim_loss = 1 - tf.image.ssim_multiscale(img1=y_true, img2=y_pred[...,:n], max_val=1.0, power_factors=power_factors)
            l1_loss = K.mean(K.abs(K.tanh(y_pred[...,:n])-K.tanh(y_true)),axis=axis)
            return compensation*(alpha * ssim_loss + l1_loss)
        return msssimhdr
    else:
        def msssimhdr(y_true, y_pred):
            n = K.shape(y_true)[1]
            axis = tuple((d for d in range(K.ndim(y_true)) if d != 0))
            ssim_loss = 1 - tf.image.ssim_multiscale(img1=y_true, img2=y_pred[:,:n,...], max_val=1.0, power_factors=power_factors)
            l1_loss = K.mean(K.abs(K.tanh(y_pred[:,:n,...])-K.tanh(y_true)),axis=axis)
            return compensation*(alpha * ssim_loss + l1_loss)
        return msssimhdr


def loss_msssiml1(mean=False):
    alpha = 0.125
    compensation= 1.0
    power_factors=(0.0448, 0.2856, 0.3001)
    if backend_channels_last():
        def msssiml1(y_true, y_pred):
            n = K.shape(y_true)[-1]
            axis = tuple((d for d in range(K.ndim(y_true)) if d != 0))
            ssim_loss = 1 - tf.image.ssim_multiscale(img1=y_true, img2=y_pred[...,:n], max_val=1.0, power_factors=power_factors)
            l1_loss = K.mean(K.abs(y_pred[...,:n] - y_true),axis=axis)
            return compensation*(alpha * ssim_loss + (1 - alpha) * l1_loss)
        return msssiml1
    else:
        def msssiml1(y_true, y_pred):
            n = K.shape(y_true)[1]
            axis = tuple((d for d in range(K.ndim(y_true)) if d != 0))
            ssim_loss = 1 - tf.image.ssim_multiscale(img1=y_true, img2=y_pred[:,:n,...], max_val=1.0, power_factors=power_factors)
            l1_loss = K.mean(K.abs(y_pred[:,:n,...] - y_true),axis=axis)
            return compensation*(alpha * ssim_loss + (1 - alpha) * l1_loss)
        return msssiml1
