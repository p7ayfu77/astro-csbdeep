from pathlib import Path

from kivy.app import App
from kivy.properties import StringProperty
from kivy.core.window import Window
from kivy.config import Config

from astrodenoise.version import version, modelversion

class AstroDeNoiseApp(App):
    def __init__(self, **kwargs):
        super(AstroDeNoiseApp, self).__init__(**kwargs)
        self.lastpath = ""  

    lastpath = StringProperty()
    lastmodel = StringProperty()

    def build(self):
        self.title = f"AstroDenoise - v{version}"
        Window.bind(on_request_close=self.window_request_close)
        config = self.config
        
        if config is not None:
            Window.left = config.getint('graphics', 'left')
            Window.top = config.getint('graphics', 'top')
            Window.size = (config.getint('graphics', 'width'), config.getint('graphics', 'height'))
            self.lastpath = config.get('astrodenoise', 'lastpath')
            self.lastmodel = config.get('astrodenoise', 'lastmodel')

        Config.set('input', 'mouse', 'mouse,disable_multitouch')

    def build_config(self, config):
        config.setdefaults('graphics', {
            'width': 1024,
            'height': 768,
            'top': 100,
            'left': 100
        })

        config.setdefaults('astrodenoise', {
            'lastpath': '',
            'lastmodel': modelversion
        })

    def window_request_close(self, win):

        config = self.config
        if config is not None:
            config.set('graphics', 'left', Window.left)
            config.set('graphics', 'top', Window.top)
            config.set('graphics', 'width', Window.system_size[0])
            config.set('graphics', 'height', Window.system_size[1])
            config.set('astrodenoise', 'lastpath', Path(self.lastpath).as_posix())
            config.set('astrodenoise', 'lastmodel', self.lastmodel)
            config.write()

        return False

def get_app(running_app) -> AstroDeNoiseApp:    
    return running_app