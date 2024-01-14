"""File dialog functions."""

import os
import pywintypes
from typing import Union, List, Optional, Tuple
import win32gui
#from win32gui import GetDesktopWindow, GetOpenFileNameW, GetSaveFileNameW, SendMessage, FindWindow
from win32com.shell import shell, shellcon
from win32con import OFN_EXPLORER, OFN_ALLOWMULTISELECT, OFN_NOCHANGEDIR
from pathlib import Path

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
        file_path, _, _ = win32gui.GetOpenFileNameW(
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
        file_path, _, _ = win32gui.GetSaveFileNameW(
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


# A callback procedure - called by SHBrowseForFolder
def BrowseCallbackProc(hwnd, msg, lp, data):
    
    if msg == shellcon.BFFM_INITIALIZED:
        address, length = win32gui.PyGetBufferAddressAndLen(data.encode("ISO8859-1"))
        win32gui.SendMessage(hwnd, shellcon.BFFM_SETSELECTION, 1, address)
    elif msg == shellcon.BFFM_SELCHANGED:
        # Set the status text of the
        # For this message, 'lp' is the address of the PIDL.
        pidl = shell.AddressAsPIDL(lp)
        try:
            path = shell.SHGetPathFromIDList(pidl)
            win32gui.SendMessage(hwnd, shellcon.BFFM_SETSTATUSTEXT, 0, path)
        except shell.error:
            # No path for this PIDL
            pass

def open_folder_dialog(title: str = "", start_folder = None, selected_folder = None, encoding: str = "ISO8859-1") -> Optional[str]:
    """Open a folder open dialog.

    :param title: Dialog title.
    :param encoding: Encoding for the folder. Default is Latin-1.
    :return: Path to folder. None if no folder selected.
    """
    # http://timgolden.me.uk/python/win32_how_do_i/browse-for-a-folder.html
    if start_folder is None:
        start_pidl = shell.SHGetFolderLocation(0, shellcon.CSIDL_DESKTOP, 0, 0)
    else:
        if Path(start_folder).exists:
            start_pidl = shell.SHParseDisplayName(start_folder, 0)[0]
        else:
            return None

    if selected_folder is not None and not Path(selected_folder).exists:
        return None

    pidl, display_name, image_list = shell.SHBrowseForFolder(
        win32gui.GetDesktopWindow(), 
        start_pidl, 
        title, 
        shellcon.BIF_STATUSTEXT | 0x0040 | shellcon.BIF_VALIDATE | shellcon.BIF_EDITBOX,
        None if selected_folder is None else BrowseCallbackProc,
        selected_folder
        )

    if (pidl, display_name, image_list) != (None, None, None):
        return shell.SHGetPathFromIDList(pidl).decode(encoding)