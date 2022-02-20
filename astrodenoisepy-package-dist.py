import sys
from cx_Freeze import setup, Executable

build_exe_options = {"build_exe":"astrodenoisepy\dist", "include_files": ["LICENSE.txt", "astrodenoisepy\dist-models\main"]}

setup(
    name = "astrodenoisepy",
    version = "0.1",
    options = {"build_exe": build_exe_options},
    description = "astrodenoisepy",
    executables = [Executable("astrodenoisepy.py")]
)