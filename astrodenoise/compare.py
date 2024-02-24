from kivy.uix.actionbar import ActionToggleButton
from kivy.event import EventDispatcher

from astrodenoise.uiimage import uiimage

class CompareExample(ActionToggleButton, EventDispatcher):
    def __init__(self, imagedata=uiimage(), **kwargs):
        super(CompareExample, self).__init__(**kwargs)
        self.imagedata:uiimage = imagedata
