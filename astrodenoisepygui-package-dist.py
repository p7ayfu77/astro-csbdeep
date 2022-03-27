import sys
from cx_Freeze import setup, Executable

build_exe_options = {
    "build_exe": "astrodenoisepy\\dist-gui",
    "packages": ["kivy"],
    "include_files": [
        "LICENSE.txt",
        ["astrodenoisepy\\dist-models\\v0.4.0-01", "models\\default"],
        ["astrodenoisepy\\dist-models\\v0.3.0-01", "models\\v0.3.0-01"],
        ["astrodenoisepy\\data", "astrodenoisepy\\data"],
        "astrodenoisepygui.kv",
        #angle
        ".venv\\share\\angle\\bin\\d3dcompiler_47.dll",
        ".venv\\share\\angle\\bin\\libEGL.dll",
        ".venv\\share\\angle\\bin\\libGLESv2.dll",
        #glew
        ".venv\\share\\glew\\bin\\glew32.dll",
        #sdl2
        ".venv\\share\\sdl2\\bin\\libFLAC-8.dll",
        #".venv\\share\\sdl2\\bin\\libfreetype-6.dll",
        ".venv\\share\\sdl2\\bin\\libjpeg-9.dll",
        ".venv\\share\\sdl2\\bin\\libmodplug-1.dll",
        ".venv\\share\\sdl2\\bin\\libmpg123-0.dll",
        ".venv\\share\\sdl2\\bin\\libogg-0.dll",
        ".venv\\share\\sdl2\\bin\\libopus-0.dll",
        ".venv\\share\\sdl2\\bin\\libopusfile-0.dll",
        ".venv\\share\\sdl2\\bin\\libpng16-16.dll",
        ".venv\\share\\sdl2\\bin\\libtiff-5.dll",
        ".venv\\share\\sdl2\\bin\\libvorbis-0.dll",
        ".venv\\share\\sdl2\\bin\\libvorbisfile-3.dll",
        ".venv\\share\\sdl2\\bin\\libwebp-7.dll",
        ".venv\\share\\sdl2\\bin\\LICENSE.FLAC.txt",
        ".venv\\share\\sdl2\\bin\\LICENSE.freetype.txt",
        ".venv\\share\\sdl2\\bin\\LICENSE.jpeg.txt",
        ".venv\\share\\sdl2\\bin\\LICENSE.modplug.txt",
        ".venv\\share\\sdl2\\bin\\LICENSE.mpg123.txt",
        ".venv\\share\\sdl2\\bin\\LICENSE.ogg-vorbis.txt",
        ".venv\\share\\sdl2\\bin\\LICENSE.opus.txt",
        ".venv\\share\\sdl2\\bin\\LICENSE.opusfile.txt",
        ".venv\\share\\sdl2\\bin\\LICENSE.png.txt",
        ".venv\\share\\sdl2\\bin\\LICENSE.tiff.txt",
        ".venv\\share\\sdl2\\bin\\LICENSE.webp.txt",
        ".venv\\share\\sdl2\\bin\\LICENSE.zlib.txt",
        ".venv\\share\\sdl2\\bin\\SDL2.dll",
        ".venv\\share\\sdl2\\bin\\SDL2_image.dll",
        ".venv\\share\\sdl2\\bin\\SDL2_mixer.dll",
        ".venv\\share\\sdl2\\bin\\SDL2_ttf.dll",
        ".venv\\share\\sdl2\\bin\\zlib1.dll",
    ]
}

import astrodenoisepyguiversion

setup(
    name="astrodenoisepygui",
    version=astrodenoisepyguiversion.version,
    options={"build_exe": build_exe_options},
    description="astrodenoisepygui",
    executables=[Executable("astrodenoisepygui.py")]
)
