import os
import sys
import argparse
import tensorflow as tf
from tifffile import imread
from xisf import XISF
import numpy as npp
from pathlib import Path
from os.path import join as path_join

from csbdeep.data import NoNormalizer, STFNormalizer, PadAndCropResizer
from csbdeep.models import CARE
from astrodeep.utils.fits import read_fits, write_fits
from astrodenoise.version import modelversion

def get_exepath():
    if getattr(sys, "frozen", False):
        datadir = os.path.dirname(sys.executable)
    else:
        datadir = Path(os.path.dirname(__file__)).parent.as_posix()
    return datadir

def cli():

    parser = argparse.ArgumentParser()

    parser.add_argument('input', type=str, nargs=1, help='Input image path, either tif or debayered fits file with data stored as 32bit float.')
    parser.add_argument('--model','-m', type=str, default=modelversion, help='Alternative model name to use for de-noising.')
    parser.add_argument('--models_folder', type=str, default='models', help='Alternative models folder root path.')
    parser.add_argument('--tiles','-t', type=int, default=3, help='Use number of tiling slices when de-noising, useful for large images and limited memory.')
    parser.add_argument('--overwrite','-o', action='store_true', help='Allow overwrite of existing output file. Default: False when not specified.')
    parser.add_argument('--device','-d', choices=['GPU','CPU'], default='CPU', help='Optional select processing to target CPU or GCP. Default: CPU')
    parser.add_argument('--normalize','-n', action='store_true', help='Enable STFNormalization before de-noising. Default: False when not specified.')
    parser.add_argument('--norm-C', type=float, default=-2.8, help='C parameter for STF Normalization. Default: -2.8')
    parser.add_argument('--norm-B', type=float, default=0.25, help='B parameter for STF Normalization, Higher B results in stronger stretch providing the ability target de-noising more effectively. . Default: 0.25, Range: 0 < B < 1')    
    parser.add_argument('--strength', type=float, default=0.5, help='The denoise strength applied. Default: 0.5')

    args = parser.parse_args()

    with tf.device(f"/{args.device}:0"):

        def predict(path,model):

            if path.suffix in ['.fit','.fits']:
                data, headers = read_fits(path)
            elif path.suffix in ['.tif','.tiff']:
                data, headers = npp.moveaxis(imread(path),-1,0), None   
            elif path.suffix in ['.xisf']:
                data, headers = npp.moveaxis(XISF(path).read_image(0),-1,0), None
            else:
                print("Skipping unsupported format. Allowed formats: .tiff/.tif/.fits/.fit")
                return

            if not npp.issubdtype(data.dtype, npp.float32):
                data = (data / npp.iinfo(data.dtype).max).astype(npp.float32)

            if data.ndim == 2:
                data = data[npp.newaxis,...]
                
            print(f"Processing file:{path}\n")
            print(f"Image Dimensions: {data.shape}\n")

            n_tiles = None if args.tiles == 0 else (args.tiles, args.tiles)
            if n_tiles is not None:
                print("Processing with tilling:",n_tiles)

            
            axes = 'YX'
            expand_low_actual = 0.5 - (args.strength/2)
            normalizer = STFNormalizer(C=args.norm_C,B=args.norm_B,expand_low=expand_low_actual,do_after=True) if args.normalize is True else NoNormalizer(expand_low=expand_low_actual,do_after=True)
            print(f"Using Normalization: {normalizer.params}\n")
            print(f"Using Strength: {args.strength}\n")
            
            output_denoised = []
            for c in data:
                output_denoised.append(
                    model.predict(c, axes, normalizer=normalizer, resizer=PadAndCropResizer(), n_tiles=n_tiles)
                    )
            
            output_denoised_arr = npp.asarray(output_denoised)
            # Clip to [0,1] range            
            output = output_denoised_arr.clip(0,1)

            output_file_name = path.stem + f"_denoised.fits"
            output_file_path = path_join(path.parent, output_file_name)
            write_fits(output_file_path, output, headers, args.overwrite)

            print("Output file saved:", output_file_path)

        print("Loading model:", args.model)
        model = CARE(config=None, name=args.model, basedir=Path(get_exepath()).joinpath(args.models_folder).as_posix())
        file_or_path = args.input[0]

        if os.path.isfile(file_or_path):
            predict(Path(file_or_path),model)
        else:
            path = Path(file_or_path)
            extensions = ('*.fits', '*.fit', '*.tiff', '*.tif')
            files_list = []
            for ext in extensions:
                files_list.extend(path.glob(ext))        
            for file in files_list:
                predict(file,model)
