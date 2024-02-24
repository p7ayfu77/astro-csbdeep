from __future__ import absolute_import, print_function
import os
import sys
import signal
import traceback
import multiprocessing

# ensure sys.stdout and sys.stderr are not None in PyInstaller environments
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")
    
def ui_start():
    os.environ['KIVY_LOG_MODE'] = 'MIXED'

    from kivy.factory import Factory
    from kivy.lang import Builder
    from kivy.app import App
    from kivy.logger import Logger

    from astrodenoise.app import get_app
    from astrodenoise.applayout import AppLayout
    from astrodenoise.dialogs import LoadDialog, LoadFolderDialog, SaveDialog
    from astrodenoise.imagelayout import ImageViewLayout
    from astrodenoise.slider import BlockingSlider

    Factory.register('Root', cls=AppLayout)
    Factory.register('LoadDialog', cls=LoadDialog)
    Factory.register('LoadFolderDialog', cls=LoadFolderDialog)
    Factory.register('SaveDialog', cls=SaveDialog)
    Factory.register('BlockingSlider', cls=BlockingSlider)
    Factory.register('BlockingSlider', cls=ImageViewLayout)

    def handle_exception(exc_type, exc_value, exc_traceback):
        """ handle all exceptions """

        # KeyboardInterrupt is a special case.
        # We don't raise the error dialog when it occurs.
        if issubclass(exc_type, KeyboardInterrupt):
            return

        Logger.error("Unhandled Exception:")
        Logger.error("".join(traceback.format_exception(exc_type, exc_value, exc_traceback)))
        try:
            get_app(App.get_running_app()).stop()        
        except Exception:
            pass

    # we want to handle TERM signal cleanly (sent by sv down)
    def handleSigTERM(a, b):
        Logger.info("Shutting Down")
        get_app(App.get_running_app()).stop()

    signal.signal(signal.SIGTERM, handleSigTERM)

    # install handler for exceptions
    sys.excepthook = handle_exception

def cli_start():    
    from astrodenoise.cli import cli
    cli()


if __name__ == '__main__':
    multiprocessing.freeze_support()
    if len(sys.argv) > 1:
        cli_start()        
    else:
        ui_start()
        from astrodenoise.app import AstroDeNoiseApp
        AstroDeNoiseApp().run()
