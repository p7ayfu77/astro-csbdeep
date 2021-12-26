from __future__ import print_function, unicode_literals, absolute_import, division

import warnings
from .care_standard import CARE

from ..utils import _raise, axes_check_and_normalize, axes_dict
from ..utils.tf import IS_TF_1, CARETensorBoardImage
from ..internals import train_hdf5
from ..data.generate_hdf5 import HDF5Data

class HDF5CARE(CARE):

    def train(self, XY_data: HDF5Data, validation_data, epochs=None, steps_per_epoch=None):
        """Train the neural network with an HDF5 data iterator of type `HDF5Data`.

        Parameters
        ----------
        XY_data : :class:`HDF5Data`
            Which returns X[i], Y[i] for XY_data[i]
        validation_data : tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`)
            Tuple of arrays for source and target validation images.
        epochs : int
            Optional argument to use instead of the value from ``config``.
        steps_per_epoch : int
            Optional argument to use instead of the value from ``config``.

        Returns
        -------
        ``History`` object
            See `Keras training history <https://keras.io/models/model/#fit>`_.

        """
        ((isinstance(validation_data,(list,tuple)) and len(validation_data)==2)
            or _raise(ValueError('validation_data must be a pair of numpy arrays')))

        n_train, n_val = len(XY_data), len(validation_data)
        frac_val = (1.0 * n_val) / (n_train + n_val)
        frac_warn = 0.05
        if frac_val < frac_warn:
            warnings.warn("small number of validation images (only %.1f%% of all images)" % (100*frac_val))
        axes = axes_check_and_normalize('S'+self.config.axes,XY_data.ndim)
        ax = axes_dict(axes)

        for a,div_by in zip(axes,self._axes_div_by(axes)):
            n = XY_data.shape[ax[a]]
            if n % div_by != 0:
                raise ValueError(
                    "training images must be evenly divisible by %d along axis %s"
                    " (which has incompatible size %d)" % (div_by,a,n)
                )

        if epochs is None:
            epochs = self.config.train_epochs
        if steps_per_epoch is None:
            steps_per_epoch = self.config.train_steps_per_epoch

        if not self._model_prepared:
            self.prepare_for_training()

        if (self.config.train_tensorboard and self.basedir is not None and
            not IS_TF_1 and not any(isinstance(cb,CARETensorBoardImage) for cb in self.callbacks)):
            self.callbacks.append(CARETensorBoardImage(model=self.keras_model, data=validation_data,
                                                       log_dir=str(self.logdir/'logs'/'images'),
                                                       n_images=3, prob_out=self.config.probabilistic))

        training_data = train_hdf5.HDF5DataWrapper(XY_data, self.config.train_batch_size, length=epochs*steps_per_epoch)
        
        fit = self.keras_model.fit_generator if IS_TF_1 else self.keras_model.fit

        history = fit(iter(training_data), validation_data=validation_data,
                      epochs=epochs, steps_per_epoch=steps_per_epoch,
                      callbacks=self.callbacks, verbose=1)
        
        self._training_finished()

        return history
