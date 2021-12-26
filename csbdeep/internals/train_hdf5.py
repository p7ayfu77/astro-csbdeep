from ..internals.train import RollingSequence
from ..data.generate_hdf5 import HDF5Data

class HDF5DataWrapper(RollingSequence):

    def __init__(self, hdf5data: HDF5Data, batch_size, length):
        self.hdf5data = hdf5data
        super(HDF5DataWrapper, self).__init__(data_size=len(hdf5data), batch_size=batch_size, length=length, shuffle=False)        

    def __getitem__(self, i):
        idx = self.batch(i)
        return self.hdf5data[idx]

    def __del__(self):
        self.hdf5data = None