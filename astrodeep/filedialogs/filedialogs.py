"""File dialog functions."""

import os
import pywintypes
from typing import Union, List, Optional, Tuple
from win32gui import GetDesktopWindow, GetOpenFileNameW, GetSaveFileNameW, FindWindow
from win32com.shell import shell, shellcon
from win32con import OFN_EXPLORER, OFN_ALLOWMULTISELECT, OFN_NOCHANGEDIR


__all__ = [
    "open_file_dialog",
    "save_file_dialog",
    "open_folder_dialog",
]


def open_file_dialog(
    title: str = None,
    directory: str = os.getcwd(),
    default_name: str = "",
    default_ext: str = "",
    ext: List[Tuple[str, str]] = None,
    multiselect: bool = False,
) -> Union[str, List[str], None]:
    """Open a file open dialog at a specified directory.

    :param title: Dialog title.
    :param directory: Directory to open file dialog in.
    :param default_name: Default file name.
    :param default_ext: Default file extension. Only letters, no dot.
    :param ext: List of available extension description + name tuples,
                e.g. [(JPEG Image, jpg), (PNG Image, png)].
    :param multiselect: Allow multiple files to be selected.
    :return: Path to a file to open if multiselect=False.
             List of the paths to files which should be opened if multiselect=True.
             None if file open dialog canceled.
    :raises IOError: File open dialog failed.
    """

    # https://programtalk.com/python-examples/win32gui.GetOpenFileNameW/

    flags = OFN_EXPLORER | OFN_NOCHANGEDIR
    if multiselect:
        flags = flags | OFN_ALLOWMULTISELECT 

    if ext is None:
        ext = "All Files\0*.*\0"
    else:
        ext = "".join([f"{name}\0{extension}\0" for name, extension in ext])

    try:
        file_path, _, _ = GetOpenFileNameW(
            InitialDir=directory,
            File=default_name,
            Flags=flags,
            Title=title,
            MaxFile=2 ** 16,
            Filter=ext,
            DefExt=default_ext,
        )

        paths = file_path.split("\0")

        if len(paths) == 1:
            return paths[0]
        else:
            for i in range(1, len(paths)):
                paths[i] = os.path.join(paths[0], paths[i])
            paths.pop(0)

        return paths

    except pywintypes.error as e:  # noqa
        if e.winerror == 0:
            return
        else:
            raise IOError()


def save_file_dialog(
    title: str = None,
    directory: str = os.getcwd(),
    default_name: str = "",
    default_ext: str = "",
    ext: List[Tuple[str, str]] = None,
) -> Optional[str]:
    """Open a file save dialog at a specified directory.

    :param title: Dialog title.
    :param directory: Directory to open file dialog in.
    :param default_name: Default file name.
    :param default_ext: Default file extension. Only letters, no dot.
    :param ext: List of available extension description + name tuples,
                e.g. [(JPEG Image, jpg), (PNG Image, png)].
    :return: Path file should be save to. None if file save dialog canceled.
    :raises IOError: File save dialog failed.
    """

    # https://programtalk.com/python-examples/win32gui.GetSaveFileNameW/

    if ext is None:
        ext = "All Files\0*.*\0"
    else:
        ext = "".join([f"{name}\0{extension}\0" for name, extension in ext])

    flags = OFN_EXPLORER | OFN_NOCHANGEDIR

    try:
        file_path, _, _ = GetSaveFileNameW(
            InitialDir=directory,
            File=default_name,
            Flags=flags,
            Title=title,
            MaxFile=2 ** 16,
            Filter=ext,
            DefExt=default_ext,
        )

        return file_path

    except pywintypes.error as e:
        if e.winerror == 0:
            return
        else:
            raise IOError()


def open_folder_dialog(title: str = "", encoding: str = "ISO8859-1") -> Optional[str]:
    """Open a folder open dialog.

    :param title: Dialog title.
    :param encoding: Encoding for the folder. Default is Latin-1.
    :return: Path to folder. None if no folder selected.
    """

    # http://timgolden.me.uk/python/win32_how_do_i/browse-for-a-folder.html

    desktop_pidl = shell.SHGetFolderLocation(0, shellcon.CSIDL_DESKTOP, 0, 0)
    pidl, display_name, image_list = shell.SHBrowseForFolder(
        GetDesktopWindow(), desktop_pidl, title, 0, None, None
    )

    if (pidl, display_name, image_list) != (None, None, None):
        return shell.SHGetPathFromIDList(pidl).decode(encoding)