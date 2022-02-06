import os
from os.path import join as path_join
from pathlib import Path
from csbdeep.data import STFNormalizer, PadAndCropResizer
from csbdeep.data.prepare import NoNormalizer
from csbdeep.models import CARE
from astrodeep.fits import read_fits, write_fits
import tensorflow as tf
import argparse
from tifffile import imread
import numpy as npp

parser = argparse.ArgumentParser()

parser.add_argument('input', type=str, nargs=1)
parser.add_argument('--model','-m', type=str, default='main')
parser.add_argument('--models_folder', type=str, default='.')
parser.add_argument('--tiles','-t', type=int, default=0)
parser.add_argument('--overwrite','-o', action='store_true')
parser.add_argument('--device','-d', choices=['GPU','CPU'], default='CPU')
parser.add_argument('--normalize','-n', action='store_true')

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

        if data.ndim == 2:
            data = data[npp.newaxis,...]
            
        print("Processing file:",path)
        print("Image Dimensions:",data.shape)

        n_tiles = None if args.tiles == 0 else (args.tiles,args.tiles)
        if n_tiles is not None:
            print("Processing with tilling:",n_tiles)

        output_denoised = []
        axes = 'YX'
        normalizer = STFNormalizer(C=-2.8,B=0.25,do_after=False) if args.normalize is True else NoNormalizer()
        for c in data:
            output_denoised.append(
                model.predict(c, axes, normalizer=normalizer,resizer=PadAndCropResizer(), n_tiles=n_tiles)
                )
        
        output_file_name = path.stem + ('_denoised_base.fits' if model.name == 'main' else f"denoised_{model.name}.fits" )
        output_path = path_join(path.parent, 'astrodenoised')
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

