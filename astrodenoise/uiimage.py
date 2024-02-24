import h5py
import tempfile
import numpy as np
import numpy.typing as npt
import shutil

    # with tempfile.NamedTemporaryFile(dir=parent_path) as temporaryFile:
    #     temporaryFileName = temporaryFile.name
h5py.get_config().track_order = True

class uiimage(object):

    def __init__(self):
        with tempfile.NamedTemporaryFile() as temporaryFile:
            temporaryFileName = temporaryFile.name
        
        self.cachefile = temporaryFileName
    
    def get_data(self, dataset):
        with h5py.File(self.cachefile, 'a', libver='latest') as f:
            if dataset in f:
                ds = f[dataset]
                assert isinstance(ds, h5py.Dataset)
                return ds[...], { k:f[dataset].attrs[k] for k in iter(f[dataset].attrs) }
            else:
                return None, {}

    def set_data(self, dataset:str, rawdata:np.ndarray, attrs: dict = {}):
        with h5py.File(self.cachefile, 'a', libver='latest') as f:
            if dataset in f:
                X = f[dataset]
            else:
                X = f.create_dataset(dataset, rawdata.shape, dtype='float32')            
            
            assert isinstance(X, h5py.Dataset)            

            X[:] = rawdata
            for k in iter(attrs):
                X.attrs[k] = attrs[k]
    
    def has_data(self, dataset):
        with h5py.File(self.cachefile, 'a', libver='latest') as f:
            if dataset in f:
                ds = f[dataset]
                assert isinstance(ds, h5py.Dataset)
                return True
        
        return False
    
    def get_clone(self):
        new = uiimage()
        shutil.copyfile(self.cachefile, new.cachefile)
        return new



