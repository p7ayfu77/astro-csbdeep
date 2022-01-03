import sys
import os
from os.path import join as path_join
from pathlib import Path
from csbdeep.data import PercentileNormalizer, PadAndCropResizer
from csbdeep.models import CARE
from astrodeep.fits import read_fits, write_fits

# Need to install csbdeep and astropy libraries

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

path = sys.argv[1]
full_model_name = sys.argv[2] if len(sys.argv) >= 3 else 'kal_p128_nPERC_PROB_spe400_e100_hdf5_main'

print("Loading model:", full_model_name)
model = CARE(config=None, name=full_model_name, basedir='models')
testaxes = 'YX'


def predict(path):
    data, headers = read_fits(path)
    output_denoised = []
    for c in data:
        # Predict/de-noise the image channel with the corresponding trained model
        # Default PercentileNormalizer is used to match the normalization used to train the model
        output_denoised.append(model.predict_probabilistic(c, testaxes, normalizer=PercentileNormalizer(),
                                                           resizer=PadAndCropResizer()).mean())
    output_file_name = path.stem + '_denoised.fits'
    output_path = path_join(path.parent, 'denoised')
    Path(output_path).mkdir(exist_ok=True)
    output_file_path = path_join(output_path, output_file_name)
    write_fits(output_file_path, output_denoised, headers)
    print("Output file saved:", output_file_path)


if os.path.isfile(path):
    predict(Path(path))
else:
    paths = Path(path).glob('*.fit*')
    for path in paths:
        predict(path)
