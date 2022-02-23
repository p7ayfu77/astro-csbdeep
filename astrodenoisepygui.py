from queue import Empty
import kivy
kivy.require('2.0.0')
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.slider import Slider
from kivy.uix.popup import Popup
from kivy.factory import Factory
from kivy.clock import mainthread
from kivy.core.window import Window
from kivy.properties import *
from kivy.graphics.texture import Texture
from kivy.logger import Logger
from kivy.clock import Clock

import os
import sys
import signal
import traceback
import threading
from pathlib import Path

import numpy as np
from tifffile import imread, imsave
from astropy.io import fits
import tensorflow as tf
from csbdeep.data import NoNormalizer, STFNormalizer, PadAndCropResizer
from csbdeep.models import CARE
from astrodeep.utils.fits import read_fits, write_fits

class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)

class SaveDialog(FloatLayout):
    save = ObjectProperty(None)
    text_input = ObjectProperty(None)
    cancel = ObjectProperty(None)

class MySlider(Slider):
    def __init__(self, **kwargs):
        self.register_event_type('on_slider_release')
        self.register_event_type('on_slider_move')
        super(MySlider, self).__init__(**kwargs)

    def on_slider_release(self):
        pass

    def on_slider_move(self):
        pass

    def on_touch_up(self, touch):
        super(MySlider, self).on_touch_up(touch)
        if touch.grab_current == self:
            self.dispatch('on_slider_release')
            return True

    def on_touch_move(self, touch):
        super(MySlider, self).on_touch_move(touch)
        if self.collide_point(*touch.pos):
            self.dispatch('on_slider_move')
            return True

class MyFloatLayout(FloatLayout):
    def __init__(self, **kwargs):
        super(MyFloatLayout, self).__init__(**kwargs)        
        self.rawimagedata = None
        self.processedimagedata = None
        self.preprocessedimagedata = None
        self.models_basedir = 'models'        
        self.models = os.listdir('models')        
        self.selected_model = 'Default'
        self.processed = False     
        self.preprocessed = False
        self.fits_headers = None
        self.process_trigger = Clock.create_trigger(self.process_now)
        self.bind(stfC=self.process_trigger, stfB=self.process_trigger,tilling=self.process_trigger,expand_low=self.process_trigger)

    imageout = ObjectProperty(None)
    filetoload = StringProperty()
    stfC = NumericProperty(-2.8)
    stfB = NumericProperty(0.25)
    tilling  = NumericProperty(2)
    sizing  = NumericProperty(1)
    expand_low = NumericProperty(0)
    denoise_enabled = BooleanProperty(False)
    normalize_enabled = BooleanProperty(True)
    autoupdate_enabled = BooleanProperty(True)
    models = ObjectProperty(None)
    selected_model = StringProperty()
    progress = NumericProperty(0)

    def show_load(self):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()    

    def show_save(self):
        content = SaveDialog(save=self.save, cancel=self.dismiss_popup)
        self._popup = Popup(title="Save file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def load_now(self):
        
        if self.filetoload == '' or self.filetoload is None:
            return

        Window.set_system_cursor('wait')

        loadthread = threading.Thread(target=self.load_callback)        
        threading.excepthook = self.load_exception_callback        
        loadthread.start()

    def load_exception_callback(self,args):
        e = args[1]
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(e).__name__, e.args)
        print (message)
        print ("".join(traceback.format_exception(args[0], args[1], args[2])))
        Window.set_system_cursor('arrow')

    def load_callback(self):
        result, headers = self.load_file_data(self.filetoload)
        self.load_result(result, headers)

    @mainthread
    def load_result(self, result, headers):
        Window.set_system_cursor('arrow')

        self.rawimagedata = result
        self.fits_headers = headers

        self.imageout = False
        self.processed = False
        self.preprocessed = False        
        self.process_now()

    def load_file_data(self, path):

        path = Path(path)

        if path.suffix in ['.fit','.fits','.fts']: 
            data, headers = read_fits(path)            
        elif path.suffix in ['.tif','.tiff']:
            data, headers = np.moveaxis(imread(path),-1,0), None            
        else:
            print("Skipping unsupported format. Allowed formats: .tiff/.tif/.fits/.fit/.fts")
            return
        
        if not np.issubdtype(data.dtype, np.float32):
            data = (data / np.iinfo(data.dtype).max).astype(np.float32)

        if data.ndim == 2:
            data = data[np.newaxis,...]

        if self.sizing != 1:
            return self.resize_image(data, self.sizing), headers

        return data, headers

    def resize_image(self,data,subsample):
        zoom_order = 1
        from scipy.ndimage.interpolation import zoom
        factor = np.ones(data.ndim)
        factor[1:3] = subsample
        return zoom(data, [1, 1/subsample, 1/subsample], mode='nearest')

    def load(self, path, filename):
        
        App.get_running_app().lastpath = path

        try:
            self.filetoload = os.path.join(path, filename[0])
        except Exception as e:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(e).__name__, e.args)
            print (message)            
            return
        finally:
            self.dismiss_popup()

        self.sizing = 1
        self.denoise_enabled = False
        self.load_now()

    def save(self, path, filename):

        self.dismiss_popup()

        filepath = Path(path,filename)

        if self.fits_headers is None:
            self.fits_headers = fits.Header()

        self.fits_headers['HISTORY'] = 'AstroDenoisePyGUI_STF_C {}'.format(self.stfC)
        self.fits_headers['HISTORY'] = 'AstroDenoisePyGUI_STF_B {}'.format(self.stfB)
        self.fits_headers['HISTORY'] = 'AstroDenoisePyGUI_STF_Low {}'.format(self.expand_low)
        self.fits_headers['HISTORY'] = 'AstroDenoisePyGUI_Model {}'.format(self.selected_model)
        self.fits_headers['HISTORY'] = 'AstroDenoisePyGUI_Scale {}'.format(self.sizing) 
        self.fits_headers['HISTORY'] = 'AstroDenoisePyGUI_Processed {}'.format(self.processed) 
        self.fits_headers['HISTORY'] = 'AstroDenoisePyGUI_AutoUpdate {}'.format(self.autoupdate_enabled)
        self.fits_headers['HISTORY'] = 'AstroDenoisePyGUI_Denoise {}'.format(self.denoise_enabled)

        if self.denoise_enabled:
            result_tosave = self.processedimagedata
            self.processed = True
        else:
            result_tosave = self.preprocessedimagedata

        try:
            if filepath.suffix in ['.fit','.fits','.fts']:
                result_forsave = np.moveaxis(np.transpose(result_tosave),1,2)
                write_fits(filepath,result_forsave,headers=self.fits_headers)
            elif filepath.suffix in ['.tif','.tiff']:
                imsave(filepath,data=result_tosave)
            else:
                print("Skipping unsupported format. Allowed formats: .tiff/.tif/.fits/.fit/.fts")
        except Exception as e:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(e).__name__, e.args)
            print (message)            

    def dismiss_popup(self):
        self._popup.dismiss()

    def on_stfB(self, instance, value):
        self.stfB = value
        self.processed = False
        self.preprocessed = False

    def on_stfC(self, instance, value):
        self.stfC = value
        self.processed = False
        self.preprocessed = False        

    def on_expand_low(self, instance, value):
        self.expand_low = value
        self.processed = False
        self.preprocessed = False        

    def on_tilling(self, instance, value):
        self.tilling = value
        self.processed = False
        self.preprocessed = False        

    def reset_sliders(self):
        self.processed = False
        self.preprocessed = False        
        self.stfC = -2.8
        self.stfB = 0.25
        self.expand_low = 0

    def denoise_check(self, instance, value):        
        self.denoise_enabled = value
        self.process_now()

    def normalize_check(self, instance, value):        
        self.normalize_enabled = value
        self.processed = False
        self.preprocessed = False        
        self.process_now()
        
    def autoupdate_check(self, instance, value):
        self.autoupdate_enabled = value        
        self.process_now()

    def update_model(self,value):
        self.selected_model = value
        self.processed = False
        self.preprocessed = False        
        self.process_now()

    def on_sizing(self, instance, value):
        self.sizing = value
        self.load_now()

    def process_now(self, *largs):
        if self.autoupdate_enabled and self.rawimagedata is not None:
            
            if self.denoise_enabled and not self.processed:
                pass
            elif self.denoise_enabled and self.processedimagedata is not None:
                self.process_result(self.processedimagedata)
                return
            elif self.denoise_enabled and self.processedimagedata is None:
                pass
            elif not self.preprocessed:
                pass
            elif self.preprocessedimagedata is not None:
                self.process_result(self.preprocessedimagedata)
                return

            Window.set_system_cursor('wait')

            processthread = threading.Thread(target=self.process_callback)
            threading.excepthook = self.process_exception_callback
            processthread.start()
    
    def process_exception_callback(self,args):
        e = args[1]
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(e).__name__, e.args)
        print (message)
        print ("".join(traceback.format_exception(args[0], args[1], args[2])))
        Window.set_system_cursor('arrow')

    def process_callback(self):
        result = self.process(self.rawimagedata,self.stfC,self.stfB)
        #(1417, 2073, 3)
        self.process_result(result)

    @mainthread
    def process_result(self,result):
        
        Window.set_system_cursor('arrow')

        if self.denoise_enabled:
            self.processedimagedata = result
            self.processed = True
        else:
            self.preprocessedimagedata = result
            self.preprocessed = True

        image = (result - np.min(result)) / (np.max(result) - np.min(result))
        image = (image * 255 / np.max(image)).astype('uint8')

        colorfmt='rgb'
        if image.shape[2] == 1:            
            colorfmt='luminance'

        texture = Texture.create(size=(image.shape[1], image.shape[0]), colorfmt=colorfmt)
        texture.blit_buffer(image.tobytes(), colorfmt=colorfmt)
        texture.flip_vertical()
        texture.mag_filter = 'nearest'
        self.imageout = texture

    @mainthread
    def update_progress(self,progress):
        self.progress = progress

    def process(self, data, C=-2.8,B=0.25):

        self.update_progress(0)

        normalizer = STFNormalizer(C=C,B=B,expand_low=self.expand_low,do_after=False) if self.normalize_enabled else NoNormalizer(expand_low=self.expand_low)

        if self.denoise_enabled:
            with tf.device(f"/GPU:0"):
                axes = 'YX'
                model = CARE(config=None, name=self.selected_model, basedir=self.models_basedir)
                output_denoised = []
                for i, c in enumerate(data):
                    output_denoised.append(
                        model.predict(c, axes, normalizer=normalizer,resizer=PadAndCropResizer(), n_tiles = (self.tilling,self.tilling))
                        )
                    self.update_progress((i+1)/len(data))
            result = np.moveaxis(np.transpose(output_denoised),0,1)
        else:
            result = normalizer.before(np.moveaxis(np.transpose(data),0,1),'YX')
            self.update_progress(1)

        return result 

class MyZoomFloatLayout(AnchorLayout):
    region_x = NumericProperty(0)
    region_y = NumericProperty(0)
    region_w = NumericProperty(0)
    region_h = NumericProperty(0)
    scale = NumericProperty(1)
    imageoutzoom = ObjectProperty(None)
    imageout = ObjectProperty(None)

    def on_imageout(self, instance, value):                
        if value:
            self.region_w = value.width / self.scale
            self.region_h = value.height / self.scale
            self.imageoutzoom.texture = value.get_region(self.region_x, self.region_y, self.region_w, self.region_h)
        else:
            self.region_w = 0
            self.region_h = 0
            self.scale = 1
            self.region_x = 0
            self.region_y = 0

        return True

    def on_touch_down(self, touch):
        if not self.collide_point(*touch.pos):
            return super().on_touch_down(touch)    

        if self.imageout is None:
            return True

        if touch.is_mouse_scrolling:
            if touch.button == 'scrolldown':
                if self.scale < 10:
                    prev_w = self.imageout.width / self.scale
                    prev_h = self.imageout.height / self.scale
                    self.scale *= 1.1
                    self.region_w = self.imageout.width / self.scale
                    self.region_h = self.imageout.height / self.scale
                    self.region_x += (prev_w-self.region_w) // 2
                    self.region_y += (prev_h-self.region_h) // 2
            elif touch.button == 'scrollup':
                if self.scale > 1:
                    prev_w = self.imageout.width / self.scale
                    prev_h = self.imageout.height / self.scale
                    self.scale /= 1.1
                    self.region_w = self.imageout.width / self.scale
                    self.region_h = self.imageout.height / self.scale
                    
                    if (self.region_w > self.imageout.width) or (self.region_h > self.imageout.height):
                        self.region_w = self.imageout.width
                        self.region_h = self.imageout.height

                    new_x = self.region_x + (prev_w-self.region_w) // 2
                    new_y = self.region_y + (prev_h-self.region_h) // 2

                    if (new_x + self.region_w) > self.imageout.width:
                        self.region_x = self.imageout.width - self.region_w
                    elif new_x < 0:
                        self.region_x = 0
                    else:
                        self.region_x += (prev_w-self.region_w) // 2

                    if (new_y + self.region_h) > self.imageout.height:
                        self.region_y = self.imageout.height - self.region_h
                    elif new_y < 0:
                        self.region_y = 0
                    else:
                        self.region_y += (prev_h-self.region_h) // 2
                else:
                    self.scale = 1
        else:
            touch.grab(self)            

        if self.imageout is not None:
            self.imageoutzoom.texture = self.imageout.get_region(self.region_x, self.region_y, self.region_w, self.region_h)
       
        return True
        #super().on_touch_down(touch)

    def on_touch_move(self, touch):
        if not self.collide_point(*touch.pos):
            return super().on_touch_move(touch)

        if touch.grab_current is self:

            imx, imy = self.children[0].get_norm_image_size()
            deltax = -touch.dx * (self.region_w/self.right) * (self.right/imx)
            deltay = -touch.dy * (self.region_h/self.top) * (self.top/imy)

            new_x = self.region_x + deltax
            new_y = self.region_y + deltay
            if (new_x >= 0) and (new_x + self.region_w <= self.imageout.width):
                self.region_x += deltax

            if (new_y >= 0) and (new_y + self.region_h <= self.imageout.height):
                self.region_y += deltay

            if self.imageout is not None:
                self.imageoutzoom.texture = self.imageout.get_region(self.region_x, self.region_y, self.region_w, self.region_h)

        return True
        #super().on_touch_move(touch)

    def on_touch_up(self, touch):
        super().on_touch_move(touch)
        
        if touch.grab_current is self:
            touch.ungrab(self)

class AstroDenoisePyGUIApp(App):
    lastpath = StringProperty()

    def build(self):
        Window.bind(on_request_close=self.window_request_close)
        
        Window.left = self.config.getint('graphics', 'left')
        Window.top = self.config.getint('graphics', 'top')
        Window.size = (self.config.getint('graphics', 'width'),self.config.getint('graphics', 'height'))

        self.lastpath = self.config.get('astrodenoisepy', 'lastpath')
        from kivy.config import Config
        Config.set('input', 'mouse', 'mouse,disable_multitouch')

    def build_config(self, config):
        config.setdefaults('graphics', {
            'height': 1024,
            'width': 1280,
            'top': 200,
            'left': 200
        })

        config.setdefaults('astrodenoisepy', {
            'lastpath': ''
        })


    def window_request_close(self, win):
        self.config.set('graphics', 'left', Window.left)
        self.config.set('graphics', 'top', Window.top)
        self.config.set('graphics', 'width', Window.size[0])
        self.config.set('graphics', 'height', Window.size[1])
        
        self.config.set('astrodenoisepy', 'lastpath', Path(self.lastpath).as_posix())

        self.config.write()

        return False    

Factory.register('Root', cls=MyFloatLayout)
Factory.register('LoadDialog', cls=LoadDialog)
Factory.register('SaveDialog', cls=SaveDialog)

def handle_exception(exc_type, exc_value, exc_traceback):
    """ handle all exceptions """

    # KeyboardInterrupt is a special case.
    # We don't raise the error dialog when it occurs.
    if issubclass(exc_type, KeyboardInterrupt):
        return

    Logger.error("Unhandled Exception:")
    Logger.error("".join(traceback.format_exception(exc_type, exc_value, exc_traceback)))
    try:
        App.get_running_app().stop()
    except Exception:
        pass


# we want to handle TERM signal cleanly (sent by sv down)
def handleSigTERM(a, b):
    App.get_running_app().stop()


signal.signal(signal.SIGTERM, handleSigTERM)

# install handler for exceptions
sys.excepthook = handle_exception

if __name__ == '__main__':
    AstroDenoisePyGUIApp().run()