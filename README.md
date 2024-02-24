# Astro CSBDeep – a toolbox for training and using denoise models for Astrophotography

**NOTE: This is a fork of [CSBDeep](https://github.com/CSBDeep/CSBDeep). Thank you to the developers.**

This is a modified version of the CSBDeep Python package, which provides a toolbox for content-aware restoration of fluorescence microscopy images (CARE), based on deep learning via Keras and TensorFlow.

Additionally some helpful changes have been added to easily process Astrophotography images.

Please see the documentation at http://csbdeep.bioimagecomputing.com/doc/.

## Environment Setup

You will need a few things to get started:

1. An NVIDIA GPU and relevant CUDA libraries installed - great video guide here [Install Tensorflow GPU and PyTorch in Windows 10 - Generalized method](https://www.youtube.com/watch?v=86Mq-h8tazc)
2. Python 3 installed
3. Create and venv environment preferabley named `.venv` by running `python -m venv .venv` and activating it `.venv\Scripts\activate`
3. Install the pip3 requirement found in the `requirements.txt` file by running `pip3 install -r requirements.txt`
4. (Optional) Install Visual Studio Code

Key Dependencies are:

* python:3.9
* [cuDNN:8.9.7](https://developer.nvidia.com/rdp/cudnn-archive)
* [CUDA:11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
* jupyter:1.0.0
* tensorflow-gpu:2.10.0

Please check the `requirements.txt` file for versions of the above and other packages used to test the example Jupyter Notebook.

## Working in Visual Studio Code

Once you have you have a python venv and dependencies are installed fire up VS Code.

Note: VS Code prefers python a venv named `.venv` by default. See above for installing python requirements.

Install the following VS Code extensions:

* python (ms-python.python)
* pylance (ms-python.vscode-pylance)
* jupyter (ms-toolsai.jupyter)
* jupyter-keymap (ms-toolsai.jupyter-keymap)

Settings in the `.vscode` folder configure the default python interpreter path for the venv folder `.venv` and your default startup terminal.

Note if you dont see the the venv startup automatically in terminal just activate as above, its temperament :) 

Open up the notebook `AstroNoise2Noise.ipynb` and configure the same python interpreter path `.venv` in the top right hand of the VS Code interface.

Happy coding and clear skies!

## Example Notebook For Astro Image Training 

The example notebook for astrophotography `AstroNoise2Noise.ipynb` can be opened in jupyter and used to step through the process.

Further details are documented in the notebook.

In your venv start up jupiter:

```
(.venv) C:\CSBDeep> jupyter notebook
```

## AstroDenoise Command Line Tool

You can use the AstroDenoise cli commands to denoise your image using your custom model.

```
(.venv) C:\CSBDeep> python -m astrodenoise.main myimage.tif
```

See the arguments help for additional options.

```
(.venv) C:\CSBDeep> python -m astrodenoise.main -h

usage: AstroDenoise.exe [-h] [--model MODEL] [--models_folder MODELS_FOLDER] [--tiles TILES] [--overwrite]
                        [--device {GPU,CPU}] [--normalize] [--norm-C NORM_C] [--norm-B NORM_B] [--norm-restore]
                        input

positional arguments:
  input                 Input image path, either tif or debayered fits file with data stored as 32bit float.

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL, -m MODEL
                        Alternative model name to use for de-noising.
  --models_folder MODELS_FOLDER
                        Alternative models folder root path.
  --tiles TILES, -t TILES
                        Use number of tiling slices when de-noising, useful for large images and limited memory.
  --overwrite, -o       Allow overwrite of existing output file. Default: False when not specified.
  --device {GPU,CPU}, -d {GPU,CPU}
                        Optional select processing to target CPU or GCP. Default: CPU
  --normalize, -n       Enable STFNormalization before de-noising. Default: False when not specified.
  --norm-C NORM_C       C parameter for STF Normalization. Default: -2.8
  --norm-B NORM_B       B parameter for STF Normalization, Higher B results in stronger stretch providing the ability
                        target de-noising more effectively. . Default: 0.25, Range: 0 < B < 1
  --norm-restore        Restores output image to original data range after processing. Default: False when not
                        specified.
```

## AstroDenoise GUI

You can use the AstroDenoise GUI to interactively denoise your image using your custom model.

Start the GUI by running the `astrodenoise.main` module entrypoint without arguments.

```
(.venv) C:\CSBDeep> python -m astrodenoise.main
```

## Contributors

Want to say a big thank you to the CSBDeep team who provided their framework along with great examples to get started.

To date, further much appreciated thanks go to contributions from:

* Michael Kitange - ongoing validation and contributions to code
* JimmyTheChicken#2719 - STF Tone Mapping Algorithm

More to come!