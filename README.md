# Astro CSBDeep – a toolbox for CARE with some tweaks

**NOTE: This is a fork of [CSBDeep](https://github.com/CSBDeep/CSBDeep). Thank you to the developers.**

This is a modified version of the CSBDeep Python package, which provides a toolbox for content-aware restoration of fluorescence microscopy images (CARE), based on deep learning via Keras and TensorFlow.

Additionally some helpful changes have been added to easily process Astrophotography images.

Please see the documentation at http://csbdeep.bioimagecomputing.com/doc/.

## Environment Setup

You will need a few things to get started:

1. An NVIDIA GPU is recommended and relevant CUDA libraries installed - great video guide here [Install Tensorflow GPU and PyTorch in Windows 10 - Generalized method](https://www.youtube.com/watch?v=86Mq-h8tazc). Astro CSBDeep can be run using only a CPU but it may be slow.
2. Python 3 installed
3. Create and venv environment preferabley named `.venv` by running `python -m venv .venv` and activating it `.venv\Scripts\activate`
3. Install the pip3 requirement found in the `requirements.txt` file by running `pip3 install -r requirements.txt`
4. (Optional) Install Visual Studio Code

Key Dependencies are:

* python:3.7-3.9
* cuDNN:8.1
* CUDA:11.2
* jupyter:1.0.0
* tensorflow-gpu:2.7.0

Please check the `requirements.txt` file for versions of the above and other packages used to test the example Jupyter Notebook.

## Example Notebook For Astro

The example notebook for astrophotography `AstroNoise2Noise.ipynb` can be opened in jupyter and used to step through the process.

Further details are documented in the notebook.

In your venv start up jupiter:

```
(.venv) C:\CSBDeep> jupyter notebook
```

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

## Running on linux
```
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python3 ./astrodenoisypygui.py
```
When the gui loads, select the denoise model v0.4.0-01. 

## Contributors

Want to say a big thank you to the CSBDeep team who provided their framework along with great examples to get started.

To date, further much appreciated thanks go to contributions from:

* Michael Kitange - ongoing validation and contributions to code
* JimmyTheChicken#2719 - STF Tone Mapping Algorithm

More to come!
