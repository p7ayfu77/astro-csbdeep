import sys
from cx_Freeze import setup, Executable

build_exe_options = {"build_exe":"dist", "include_files": ["LICENSE.txt", "dist-models\main"]}

setup(
    name = "AstroDenoisePy",
    version = "0.1",
    options = {"build_exe": build_exe_options},
    description = "AstroDenoisePy",
    executables = [Executable("AstroDenoisePy.py")]
)