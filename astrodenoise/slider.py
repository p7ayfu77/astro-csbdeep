from kivy.event import EventDispatcher
from kivy.uix.slider import Slider

class BlockingSlider(Slider,EventDispatcher):
    def __init__(self, **kwargs):        
        super(BlockingSlider, self).__init__(**kwargs)
        self.register_event_type('on_slider_release')
        self.register_event_type('on_slider_move')

    def on_slider_release(self):
        pass

    def on_slider_move(self):
        pass

    def on_touch_up(self, touch):
        super(BlockingSlider, self).on_touch_up(touch)
        if touch.grab_current == self:
            self.dispatch('on_slider_release')
            return True

    def on_touch_move(self, touch):
        super(BlockingSlider, self).on_touch_move(touch)
        if self.collide_point(*touch.pos):
            self.dispatch('on_slider_move')
            return True
