from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter


from ..utils import _raise, consume, normalize_mi_ma, axes_dict, axes_check_and_normalize, move_image_axes
import warnings
import numpy as np


from six import add_metaclass
from abc import ABCMeta, abstractmethod, abstractproperty
from scipy import stats


@add_metaclass(ABCMeta)
class Normalizer():
    """Abstract base class for normalization methods."""

    @abstractmethod
    def before(self, x, axes):
        """Normalization of the raw input image (method stub).

        Parameters
        ----------
        x : :class:`numpy.ndarray`
            Raw input image.
        axes : str
            Axes of input image x

        Returns
        -------
        :class:`numpy.ndarray`
            Normalized input image with suitable values for neural network input.
        """

    @abstractmethod
    def after(self, mean, scale, axes):
        """Possible adjustment of predicted restored image (method stub).

        Parameters
        ----------
        mean : :class:`numpy.ndarray`
            Predicted restored image or per-pixel ``mean`` of Laplace distributions
            for probabilistic model.
        scale: :class:`numpy.ndarray` or None
            Per-pixel ``scale`` of Laplace distributions for probabilistic model (``None`` otherwise.)
        axes : str
            Axes of ``mean`` and ``scale``

        Returns
        -------
        :class:`numpy.ndarray`
            Adjusted restored image(s).
        """

    def __call__(self, x, axes):
        """Alias for :func:`before` to make this callable."""
        return self.before(x, axes)

    @abstractproperty
    def do_after(self):
        """bool : Flag to indicate whether :func:`after` should be called."""


class NoNormalizer(Normalizer):
    """No normalization.

    Parameters
    ----------
    do_after : bool
        Flag to indicate whether to undo normalization.

    Raises
    ------
    ValueError
        If :func:`after` is called, but parameter `do_after` was set to ``False`` in the constructor.
    """

    def __init__(self, do_after=False):
        self._do_after = do_after

    def before(self, x, axes):
        return x

    def after(self, mean, scale, axes):
        self.do_after or _raise(ValueError())
        return mean, scale

    @property
    def do_after(self):
        return self._do_after


class PercentileNormalizer(Normalizer):
    """Percentile-based image normalization.

    Parameters
    ----------
    pmin : float
        Low percentile.
    pmax : float
        High percentile.
    do_after : bool
        Flag to indicate whether to undo normalization (original data type will not be restored).
    dtype : type
        Data type after normalization.
    kwargs : dict
        Keyword arguments for :func:`csbdeep.utils.normalize_mi_ma`.
    """

    def __init__(self, pmin=2, pmax=99.8, do_after=True, dtype=np.float32, **kwargs):
        """TODO."""
        (np.isscalar(pmin) and np.isscalar(pmax) and 0 <= pmin < pmax <= 100) or _raise(ValueError())
        self.pmin = pmin
        self.pmax = pmax
        self._do_after = do_after
        self.dtype = dtype
        self.kwargs = kwargs

    def before(self, x, axes):
        """Percentile-based normalization of raw input image.

        See :func:`csbdeep.predict.Normalizer.before` for parameter descriptions.
        Note that percentiles are computed individually for each channel (if present in `axes`).
        """
        self.axes_before = axes_check_and_normalize(axes,x.ndim)
        axis = tuple(d for d,a in enumerate(self.axes_before) if a != 'C')
        self.mi = np.percentile(x,self.pmin,axis=axis,keepdims=True).astype(self.dtype,copy=False)
        self.ma = np.percentile(x,self.pmax,axis=axis,keepdims=True).astype(self.dtype,copy=False)
        return normalize_mi_ma(x, self.mi, self.ma, dtype=self.dtype, **self.kwargs)

    def after(self, mean, scale, axes):
        """Undo percentile-based normalization to map restored image to similar range as input image.

        See :func:`csbdeep.predict.Normalizer.after` for parameter descriptions.

        Raises
        ------
        ValueError
            If parameter `do_after` was set to ``False`` in the constructor.

        """
        self.do_after or _raise(ValueError())
        self.axes_after = axes_check_and_normalize(axes,mean.ndim)
        mi = move_image_axes(self.mi, self.axes_before, self.axes_after, True)
        ma = move_image_axes(self.ma, self.axes_before, self.axes_after, True)
        alpha = ma - mi
        beta  = mi
        return (
            ( alpha*mean+beta ).astype(self.dtype,copy=False),
            ( alpha*scale     ).astype(self.dtype,copy=False) if scale is not None else None
        )

    @property
    def do_after(self):
        """``do_after`` parameter from constructor."""
        return self._do_after


class ReinhardNormalizer(Normalizer):

    def __init__(self, do_after=False):
        self._do_after = do_after

    def before(self, x, axes):
        return pow(x/(1+x),1/2.2)

    def after(self, mean, scale, axes):
        self.do_after or _raise(ValueError())
        return mean, scale

    @property
    def do_after(self):
        return self._do_after


class STFNormalizer(Normalizer):

    def __init__(self, C=-4.0, B=0.185, do_after=False):
        self._do_after = do_after
        self.normalizer = STFPreProcessor(C=C,B=B)

    def before(self, x, axes):
        return self.normalizer.before(x,axes)

    def after(self, mean, scale, axes):
        self.do_after or _raise(ValueError())
        return mean, scale

    @property
    def do_after(self):
        return self._do_after


@add_metaclass(ABCMeta)
class Resizer():
    """Abstract base class for resizing methods."""

    @abstractmethod
    def before(self, x, axes, axes_div_by):
        """Resizing of the raw input image (method stub).

        Parameters
        ----------
        x : :class:`numpy.ndarray`
            Raw input image.
        axes : str
            Axes of input image x
        axes_div_by : iterable of int
            Resized image must be evenly divisible by the provided values for each axis.

        Returns
        -------
        :class:`numpy.ndarray`
            Resized input image.
        """

    @abstractmethod
    def after(self, x, axes):
        """Resizing of the restored image (method stub).

        Parameters
        ----------
        x : :class:`numpy.ndarray`
            Restored image.
        axes : str
            Axes of restored image x

        Returns
        -------
        :class:`numpy.ndarray`
            Resized restored image.
        """


class NoResizer(Resizer):
    """No resizing.

    Raises
    ------
    ValueError
        In :func:`before`, if image resizing is necessary.
    """

    def before(self, x, axes, axes_div_by):
        axes = axes_check_and_normalize(axes,x.ndim)
        consume (
            (s%div_n==0) or _raise(ValueError('%d (axis %s) is not divisible by %d.' % (s,a,div_n)))
            for a, div_n, s in zip(axes, axes_div_by, x.shape)
        )
        return x

    def after(self, x, axes):
        return x


class PadAndCropResizer(Resizer):
    """Resize image by padding and cropping.

    If necessary, input image is padded before prediction
    and restored image is cropped back to size of input image
    after prediction.

    Parameters
    ----------
    mode : str
        Parameter ``mode`` of :func:`numpy.pad` that
        controls how the image is padded.
    kwargs : dict
        Keyword arguments for :func:`numpy.pad`.
    """

    def __init__(self, mode='reflect', **kwargs):
        """TODO."""
        self.mode = mode
        self.kwargs = kwargs

    def before(self, x, axes, axes_div_by):
        """Pad input image.

        See :func:`csbdeep.predict.Resizer.before` for parameter descriptions.
        """
        axes = axes_check_and_normalize(axes,x.ndim)
        def _split(v):
            a = v // 2
            return a, v-a
        self.pad = {
            a : _split((div_n-s%div_n)%div_n)
            for a, div_n, s in zip(axes, axes_div_by, x.shape)
        }
        # print(self.pad)
        x_pad = np.pad(x, tuple(self.pad[a] for a in axes), mode=self.mode, **self.kwargs)
        return x_pad

    def after(self, x, axes):
        """Crop restored image to retain size of input image.

        See :func:`csbdeep.predict.Resizer.after` for parameter descriptions.
        """
        axes = axes_check_and_normalize(axes,x.ndim)
        all(a in self.pad for a in axes) or _raise(ValueError())
        crop = tuple (
            slice(p[0], -p[1] if p[1]>0 else None)
            for p in (self.pad[a] for a in axes)
        )
        # print(crop)
        return x[crop]


from colour_hdri import tonemapping_operator_Reinhard2004

@add_metaclass(ABCMeta)
class PreProcessor():
    """Abstract base class for Pre-processor methods."""

    @abstractmethod
    def before(self, x, axes):
        """Pre-processing of the raw input image (method stub).

        Parameters
        ----------
        x : :class:`numpy.ndarray`
            Raw input image.
        axes : str
            Axes of input image x

        Returns
        -------
        :class:`numpy.ndarray`
            Pre-processed input image with suitable values for neural network input.
        """

    @abstractmethod
    def after(self, mean, scale, axes):
        """Possible adjustment of predicted restored image (method stub).

        Parameters
        ----------
        mean : :class:`numpy.ndarray`
            Predicted restored image or per-pixel ``mean`` of Laplace distributions
            for probabilistic model.
        scale: :class:`numpy.ndarray` or None
            Per-pixel ``scale`` of Laplace distributions for probabilistic model (``None`` otherwise.)
        axes : str
            Axes of ``mean`` and ``scale``

        Returns
        -------
        :class:`numpy.ndarray`
            Adjusted restored image(s).
        """

    def __call__(self, x, axes):
        """Alias for :func:`before` to make this callable."""
        return self.before(x, axes)

    @abstractproperty
    def do_after(self):
        """bool : Flag to indicate whether :func:`after` should be called."""


class NoPreProcessor(PreProcessor):
    """No Pre-processing.

    Parameters
    ----------
    do_after : bool
        Flag to indicate whether to undo normalization.

    Raises
    ------
    ValueError
        If :func:`after` is called, but parameter `do_after` was set to ``False`` in the constructor.
    """

    def __init__(self, do_after=False):
        self._do_after = do_after

    def before(self, x, axes):
        return x

    def after(self, mean, scale, axes):
        self.do_after or _raise(ValueError())
        return mean, scale

    @property
    def do_after(self):
        return self._do_after


class ReinhardPreProcessor(PreProcessor):
    """No Pre-processing.

    Parameters
    ----------
    do_after : bool
        Flag to indicate whether to undo normalization.

    Raises
    ------
    ValueError
        If :func:`after` is called, but parameter `do_after` was set to ``False`` in the constructor.
    """

    def __init__(self, do_after=False, **kwargs):
        self._do_after = do_after
        self.kwargs = kwargs

    def before(self, x, axes):
        return tonemapping_operator_Reinhard2004(x, **self.kwargs)

    def after(self, mean, scale, axes):
        self.do_after or _raise(ValueError())
        return mean, scale

    @property
    def do_after(self):
        return self._do_after

import numexpr
class STFPreProcessor(PreProcessor):
    """No Pre-processing.

    Parameters
    ----------
    do_after : bool
        Flag to indicate whether to undo normalization.

    Raises
    ------
    ValueError
        If :func:`after` is called, but parameter `do_after` was set to ``False`` in the constructor.
    """
    
    @staticmethod
    def mtf(m,x):
        # From: https://pixinsight.com/forum/index.php?threads/mathematically-exact-inverse-of-mtf.14912/
        
        # This seems to be slower
        # try:
        #     import numexpr
        #     mtf_result = numexpr.evaluate("((m - 1.0) * x) / ( (2.0 * m - 1.0) * x - m )",casting='no')
        # except ImportError:
        #     mtf_result =                   ((m - 1.0) * x) / ( (2.0 * m - 1.0) * x - m )
        
        # return mtf_result
        
        n = (m - 1.0) * x
        d = (2.0 * m - 1.0) * x - m
        return n / d        

    @staticmethod
    def stf(d,C=-4.0,B=0.185,axis=[0,1]):
        # From: JimmyTheChicken#2719 nightlyastronomy.com
        # C = -2.8
        # B = 0.25
        # c = min( max( 0, med( $T ) + C*mdev( $T ) ), 1 );
        # mtf( mtf( B, med( $T ) - c ), max( 0, ($T - c)/~c ) )
        
        med = np.median(d,axis=axis)
        #mad = stats.median_absolute_deviation(d,axis=axis,scale=1.4826)
        mad = stats.median_abs_deviation(d,axis=axis,scale='normal')
        c = np.clip(med + C*mad,a_min=0,a_max=1)
        m = STFPreProcessor.mtf(B, med - c)
        t = np.clip((d - c)/(1.0 - c),a_min=0,a_max=1)
        return STFPreProcessor.mtf(m, t)

    @staticmethod
    def mad(arr,axis):
        """ Median Absolute Deviation: a "Robust" version of standard deviation.
            Indices variabililty of the sample.
            https://en.wikipedia.org/wiki/Median_absolute_deviation 
        """
        #arr = np.ma.array(arr).compressed() # should be faster to not use masked arrays.
        med = np.median(arr, axis=axis)
        return np.median(np.abs(arr - med),axis=axis)

    def __init__(self, C=-4.0, B=0.185, do_after=False, **kwargs):
        self._do_after = do_after
        self.kwargs = kwargs
        self.C = C
        self.B = B

    def before(self, x, axes):
        axis = tuple(d for d,a in enumerate(axes) if a != 'C')
        return STFPreProcessor.stf(x, C=self.C, B=self.B, axis=axis)

    def after(self, mean, scale, axes):
        self.do_after or _raise(ValueError())
        return mean, scale

    @property
    def do_after(self):
        return self._do_after
