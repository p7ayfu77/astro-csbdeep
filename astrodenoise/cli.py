import os
import argparse
import tensorflow as tf
from tifffile import imread
import numpy as npp
from pathlib import Path
from os.path import join as path_join

from csbdeep.data import NoNormalizer, STFNormalizer, PadAndCropResizer
from csbdeep.models import CARE
from astrodeep.utils.fits import read_fits, write_fits
from astrodenoise.version import modelversion

def cli():

    parser = argparse.ArgumentParser()

    parser.add_argument('input', type=str, nargs=1, help='Input image path, either tif or debayered fits file with data stored as 32bit float.')
    parser.add_argument('--model','-m', type=str, default=modelversion, help='Alternative model name to use for de-noising.')
    parser.add_argument('--models_folder', type=str, default='models', help='Alternative models folder root path.')
    parser.add_argument('--tiles','-t', type=int, default=0, help='Use number of tiling slices when de-noising, useful for large images and limited memory.')
    parser.add_argument('--overwrite','-o', action='store_true', help='Allow overwrite of existing output file. Default: False when not specified.')
    parser.add_argument('--device','-d', choices=['GPU','CPU'], default='CPU', help='Optional select processing to target CPU or GCP. Default: CPU')
    parser.add_argument('--normalize','-n', action='store_true', help='Enable STFNormalization before de-noising. Default: False when not specified.')
    parser.add_argument('--norm-C', type=float, default=-2.8, help='C parameter for STF Normalization. Default: -2.8')
    parser.add_argument('--norm-B', type=float, default=0.25, help='B parameter for STF Normalization, Higher B results in stronger stretch providing the ability target de-noising more effectively. . Default: 0.25, Range: 0 < B < 1')
    parser.add_argument('--norm-restore', action='store_true', help='Restores output image to original data range after processing. Default: False when not specified.')

    args = parser.parse_args()

    with tf.device(f"/{args.device}:0"):

        def predict(path,model):

            if path.suffix in ['.fit','.fits']:
                data, headers = read_fits(path)
            elif path.suffix in ['.tif','.tiff']:
                data, headers = npp.moveaxis(imread(path),-1,0), None            
            else:
                print("Skipping unsupported format. Allowed formats: .tiff/.tif/.fits/.fit")
                return

            if not npp.issubdtype(data.dtype, npp.float32):
                data = (data / npp.iinfo(data.dtype).max).astype(npp.float32)

            if data.ndim == 2:
                data = data[npp.newaxis,...]
                
            print("Processing file:",path)
            print("Image Dimensions:",data.shape)

            n_tiles = None if args.tiles == 0 else (args.tiles,args.tiles)
            if n_tiles is not None:
                print("Processing with tilling:",n_tiles)

            output_denoised = []
            axes = 'YX'
            normalizer = STFNormalizer(C=args.norm_C,B=args.norm_B,do_after=args.norm_restore) if args.normalize is True else NoNormalizer()
            print("Using Normalization:",normalizer.params)
            
            for c in data:
                output_denoised.append(
                    model.predict(c, axes, normalizer=normalizer,resizer=PadAndCropResizer(), n_tiles=n_tiles)
                    )
            
            output_file_name = path.stem + f"_denoised.fits"
            output_path = path_join(path.parent, 'denoised')
            Path(output_path).mkdir(exist_ok=True)
            output_file_path = path_join(output_path, output_file_name)
            write_fits(output_file_path, output_denoised, headers, args.overwrite)
            print("Output file saved:", output_file_path)


        print("Loading model:", args.model)
        model = CARE(config=None, name=args.model, basedir=args.models_folder)    
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
