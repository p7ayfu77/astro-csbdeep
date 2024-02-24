import os
import sys
import time
import tqdm
import threading
import traceback
from pathlib import Path
import numpy as np
from tifffile import imread, imsave
from astropy.io import fits

import kivy
kivy.require('2.3.0')
from kivy.uix.popup import Popup
from kivy.core.window import Window
from kivy.app import App
from kivy.event import EventDispatcher
from kivy.uix.floatlayout import FloatLayout
from kivy.clock import Clock
from kivy.properties import ObjectProperty, StringProperty, NumericProperty, BooleanProperty
from kivy.clock import mainthread
from kivy.graphics.texture import Texture

from astrodeep.utils.fits import read_fits, write_fits
from csbdeep.data import NoNormalizer, STFNormalizer, PadAndCropResizer
from csbdeep.models import HDF5CARE

import astrodenoise
from astrodenoise.version import modelversion
from astrodenoise.dialogs import LoadFolderDialog, LoadDialog, SaveDialog
from astrodenoise.compare import CompareExample
from astrodenoise.app import get_app
from astrodenoise.uiimage import uiimage

import tensorflow as tf

def isWindows():
    return sys.platform == 'win32'

supported_save_formats_fits = ['.fit','.fits','.fts']
supported_save_formats_tiff = ['.tif','.tiff']
file_dialog_ext = [
                ("Supported Image", "*.fit;*.fits;*.fts;*.FITS;*.FIT;*.FTS;*.tiff;*.tif;*.TIF;*.TIFF"),
                ("TIF Image", "*.tiff;*.tif;*.TIF;*.TIFF"),
                ("FITS Image", "*.fit;*.fits;*.fts;*.FITS;*.FIT;*.FTS")
]

class AppLayout(FloatLayout,EventDispatcher):
    def __init__(self, **kwargs):
        super(AppLayout, self).__init__(**kwargs)        
        self.currentimage: uiimage = uiimage()
        self.models_basedir = 'models'        
        self.models = os.listdir('models')        
        self.processed = False     
        self.preprocessed = False
        self.fits_headers = None
        self.trigger_process = Clock.create_trigger(self.process_triggered)
        self.bind(stfC=self.trigger_process, stfB=self.trigger_process,tilling=self.trigger_process,expand_low=self.trigger_process)
        self.stop_event = threading.Event()
        self.processthread = None
        self.iconpath = Path(astrodenoise.__file__).parent.relative_to(Path.cwd()).as_posix()
        self.trigger_onstart = Clock.create_trigger(self.on_start)
        Clock.schedule_once(self.on_start)        
    
    toolspanel = ObjectProperty(None)
    iconpath = StringProperty()
    imageout = ObjectProperty(None, allownone=True)
    imageorig = ObjectProperty(None, allownone=True)
    outparams = ObjectProperty(None, allownone=True)
    origparams = ObjectProperty(None, allownone=True)
    filetoload = StringProperty()
    stfC = NumericProperty(-2.8)
    stfB = NumericProperty(0.25)
    tilling  = NumericProperty(3)
    sizing  = NumericProperty(1)
    expand_low = NumericProperty(0.5)
    denoise_enabled = BooleanProperty(False)
    normalize_enabled = BooleanProperty(True)
    autoupdate_enabled = BooleanProperty(True)
    models = ObjectProperty(None)
    selected_model = StringProperty()
    progress = NumericProperty(0)
    selected_device = StringProperty('GPU')
    snapshotinfo = StringProperty()
    modelinfo = StringProperty()

    def on_start(self, *largs):
        lastmodel = get_app(App.get_running_app()).lastmodel
        if not lastmodel:
            self.selected_model = modelversion
        else:
            self.selected_model = lastmodel

    def dismiss_popup(self):
        self._popup.dismiss()

    def select_model(self):
        models_path = Path(os.getcwd()).joinpath("models")

        if isWindows():
            from astrodeep.filedialogs import open_folder_dialog
            result = open_folder_dialog(
                "Select Model folder",
                start_folder=str(models_path),
                selected_folder=str(models_path.joinpath(self.selected_model)))
                
            if result is None:
                return

            self.selected_model = Path(result).relative_to(models_path).as_posix()
            get_app(App.get_running_app()).lastmodel = self.selected_model
            self.processed = False
            #self.preprocessed = False        
            self.trigger_process()
        else:
            content = LoadFolderDialog(load=self.load_folder, cancel=self.dismiss_popup)
            self._popup = Popup(title="Load folder", content=content,
                                size_hint=(0.9, 0.9))
            self._popup.open()    

    def load_folder(self, path):

        if (isinstance(path,list) and len(path) == 0) or path is None:
            return

        if isinstance(path,list):
            path = path[0]

        try:
            if not Path(path).exists() or not Path(path).is_dir():
                raise Exception(f"Path not valid folder: {path}")
            
            models_path = Path(os.getcwd()).joinpath("models")

            self.selected_model = Path(path).relative_to(models_path).as_posix()
            get_app(App.get_running_app()).lastmodel = self.selected_model
            self.processed = False
            #self.preprocessed = False        
            self.trigger_process()
            
        except Exception as e:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(e).__name__, e.args)
            print (message)            
            return
        finally:
            if not isWindows():
                self.dismiss_popup()

    def show_load(self):
        
        if isWindows():
            from astrodeep.filedialogs import open_file_dialog
            path = open_file_dialog(
                "Select FITS or TIFF image",
                get_app(App.get_running_app()).lastpath,
                ext=file_dialog_ext)

            if path is None:
                return

            self.load(path) 
        else:
            content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
            self._popup = Popup(title="Load file", content=content,
                                size_hint=(0.9, 0.9))
            self._popup.open()    
      
    def load(self, path):
        
        if (isinstance(path,list) and len(path) == 0) or path is None:
            return

        if isinstance(path,list):
            path = path[0]

        get_app(App.get_running_app()).lastpath = str(Path(path).parent)

        try:
            exists = Path(path).exists()
            if not exists:
                raise Exception(f"Path does not exists: {path}")
            self.filetoload = path
        except Exception as e:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(e).__name__, e.args)
            print (message)            
            return
        finally:
            if not isWindows():
                self.dismiss_popup()

        snapshots = self.ids.av
        for child in snapshots.children[:]:
            if isinstance(child,CompareExample):
                snapshots.remove_widget(child)
        self.snapshotinfo = ""

        self.sizing = 1        
        self.load_now()

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
        if result is None:
            return
        self.load_result(result, headers)

    @mainthread
    def load_result(self, result, headers):
        Window.set_system_cursor('arrow')

        self.currentimage = uiimage()
        self.currentimage.set_data('raw',result)

        self.fits_headers = headers

        self.imageout = None
        self.imageorig = None
        self.processed = False
        self.preprocessed = False        
        self.denoise_enabled = False
        self.trigger_process()

    def load_file_data(self, path):

        path = Path(path)
        extension = path.suffix.lower()

        if extension in supported_save_formats_fits: 
            data, headers = read_fits(path)            
        elif extension in supported_save_formats_tiff:
            data, headers = np.moveaxis(imread(path),-1,0), None            
        else:
            print("Skipping unsupported format. Allowed formats: .tiff/.tif/.fits/.fit/.fts")
            return None, None
        
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

    def show_save(self):

        if isWindows():
            from astrodeep.filedialogs import save_file_dialog
            path = save_file_dialog(
                        "Save FITS or TIFF image",
                        get_app(App.get_running_app()).lastpath,
                        ext=file_dialog_ext)

            if path is None:
                return

            self.save(path)         
        else:
            content = SaveDialog(save=self.save, cancel=self.dismiss_popup)
            self._popup = Popup(title="Save file", content=content,
                                size_hint=(0.9, 0.9))
            self._popup.open()

    def save(self, path):

        if not isWindows():
            self.dismiss_popup()
        
        filepath = Path(path)
        #AstroDeNoiseApp()
        get_app(App.get_running_app()).lastpath = str(filepath.parent)

        if self.fits_headers is None:
            self.fits_headers = fits.Header()

        self.fits_headers['HISTORY'] = 'AstroDenoise_STF_C {}'.format(self.stfC)
        self.fits_headers['HISTORY'] = 'AstroDenoise_STF_B {}'.format(self.stfB)
        self.fits_headers['HISTORY'] = 'AstroDenoise_STF_Low {}'.format(self.expand_low)
        self.fits_headers['HISTORY'] = 'AstroDenoise_Model {}'.format(self.selected_model)
        self.fits_headers['HISTORY'] = 'AstroDenoise_Scale {}'.format(self.sizing) 
        self.fits_headers['HISTORY'] = 'AstroDenoise_Processed {}'.format(self.processed) 
        self.fits_headers['HISTORY'] = 'AstroDenoise_AutoUpdate {}'.format(self.autoupdate_enabled)
        self.fits_headers['HISTORY'] = 'AstroDenoise_Denoise {}'.format(self.denoise_enabled)
        
        if self.denoise_enabled:            
            result_tosave = np.array(self.currentimage.get_data('post')[0])
            self.processed = True
        else:
            result_tosave = np.array(self.currentimage.get_data('pre')[0])

        try:
            extension = filepath.suffix.lower()

            if extension in supported_save_formats_fits:
                result_forsave = np.moveaxis(np.transpose(result_tosave),1,2)
                write_fits(filepath,result_forsave,headers=self.fits_headers)
            elif extension in supported_save_formats_tiff:
                result_tosave = (result_tosave - np.min(result_tosave)) / (np.max(result_tosave) - np.min(result_tosave))
                result_tosave = (result_tosave * np.iinfo(np.uint16).max).astype(np.uint16)
                imsave(filepath,data=result_tosave)
            else:
                print("Skipping unsupported format. Allowed formats: .tiff/.tif/.fits/.fit/.fts")
        except Exception as e:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(e).__name__, e.args)
            print (message)            

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
        #self.preprocessed = False        

    def on_tilling(self, instance, value):
        self.tilling = value
        self.processed = False
        #self.preprocessed = False        

    def reset_sliders(self):
        self.processed = False
        #self.preprocessed = False
        self.stfC = -2.8
        self.stfB = 0.25
        #self.expand_low = 0.9

    def denoise_check(self, instance, value):
        self.denoise_enabled = value
        self.trigger_process()

    def normalize_check(self, instance, value):
        self.normalize_enabled = value
        self.processed = False
        self.preprocessed = False        
        self.trigger_process()
        
    def autoupdate_check(self, instance, value):
        self.autoupdate_enabled = value        
        self.trigger_process()

    # def on_selected_model(self, instance, value):
        
    #     if value:
    #         model = HDF5CARE(config=None, name=value, basedir=self.models_basedir)
    #         from pprint import pformat
    #         self.modelinfo = pformat(vars(model.config), depth=5, width=3)

    def update_model(self,value):
        self.selected_model = value
        self.processed = False
        #self.preprocessed = False        
        self.trigger_process()

    def update_device(self,value):
        self.selected_device = value
        self.processed = False
        #self.preprocessed = False
        self.trigger_process()

    def on_sizing(self, instance, value):
        self.sizing = value
        self.load_now()

    def process_triggered(self, *largs):
                
        if not self.autoupdate_enabled and len(largs) != 0:
            return

        if not self.currentimage.has_data('raw'):
            return
       
        haspostdata = self.currentimage.has_data('post')
        haspredata = self.currentimage.has_data('pre')

        if self.denoise_enabled and not self.processed:
            pass
        elif self.denoise_enabled and haspostdata:
            self.processed = True
            self.display_result()
            return
        elif self.denoise_enabled and not haspostdata:
            pass
        elif not self.preprocessed:
            pass
        elif haspredata:
            self.preprocessed = True
            self.display_result()
            return
        
        if self.processthread is not None and self.processthread.is_alive():
            self.stop_event.set()            
            with tqdm.tqdm(desc='Waiting for thread to stop...') as waiter_loger:
                wait_count = 0
                while self.processthread.is_alive():
                    wait_count += 1
                    time.sleep(0.1)
                    waiter_loger.update(wait_count)

        self.stop_event.clear() 

        Window.set_system_cursor('wait')

        self.processthread = threading.Thread(target=self.execute_process)
        threading.excepthook = self.execute_process_exception
        self.processthread.start()
    
    def execute_process_exception(self,args):
        e = args[1]
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(e).__name__, e.args)
        print (message)
        print ("".join(traceback.format_exception(args[0], args[1], args[2])))
        Window.set_system_cursor('arrow')

    def execute_process(self):
        result = self.denoise(self.currentimage.get_data('raw')[0],self.stfC,self.stfB)
        
        if result is not None:
            self.process_result(result)

    def denoise(self, data, C=-2.8,B=0.25):

        self.update_progress(0)
        expand_low_actual = 0.5 - (self.expand_low/2)
        normalizer = STFNormalizer(C=C,B=B,expand_low=expand_low_actual,do_after=False) if self.normalize_enabled else NoNormalizer(expand_low=expand_low_actual)

        if self.denoise_enabled:
            with tf.device(f"/{self.selected_device}:0"):
                axes = 'YX'
                model = HDF5CARE(config=None, name=self.selected_model, basedir=self.models_basedir)
                output_denoised = []
                for i, c in enumerate(data):
                    if self.stop_event.is_set():
                        return None
                    output_denoised.append(
                        model.predict(c, axes, normalizer=normalizer,resizer=PadAndCropResizer(), n_tiles = (self.tilling,self.tilling))
                    )
                    self.update_progress((i+1)/len(data))
                result = np.moveaxis(np.transpose(output_denoised),0,1)
        else:
            result = normalizer.before(np.moveaxis(np.transpose(data),0,1),'YX')
            self.update_progress(1)

        return result 

    @mainthread
    def update_progress(self,progress):
        self.progress = progress

    @mainthread
    def process_result(self, result: np.ndarray):

        Window.set_system_cursor('arrow')
        
        if self.denoise_enabled:
            self.currentimage.set_data('post', result, self.get_label_data())
            self.processed = True
        else:
            self.currentimage.set_data('pre', result, self.get_label_data())
            self.preprocessed = True

        self.display_result()
    
    def display_result(self):

        dataset = "post" if self.denoise_enabled else "pre"
        result, labels = self.currentimage.get_data(dataset)

        self.imageout = self.get_texture(result)
        self.outparams = labels
        
        if self.denoise_enabled:
            pre, labels = self.currentimage.get_data('pre')
            if pre is not None:
                self.imageorig = self.get_texture(pre)
                self.origparams = labels
            else:
                self.imageorig = None
                self.origparams = {}

    def get_label_data(self):
        return {
            "Pre-process": "STF" if self.normalize_enabled else "Disabled",
            "STF.C": self.stfC if self.normalize_enabled else "Disabled",
            "STF.B": self.stfB if self.normalize_enabled else "Disabled",
            "Denoise": self.denoise_enabled,
            "Denoise.Strength": self.expand_low if self.denoise_enabled else "Disabled",
            "Model": self.selected_model if self.denoise_enabled else "None"
        }

    def get_texture(self, result):
        image = (result - np.min(result)) / (np.max(result) - np.min(result))
        image = (image * 255).astype('uint8')

        colorfmt='rgb'
        if image.shape[2] == 1:            
            colorfmt='luminance'

        texture = Texture.create(size=(image.shape[1], image.shape[0]), colorfmt=colorfmt)
        texture.blit_buffer(image.tobytes(), colorfmt=colorfmt)
        texture.flip_vertical()
        texture.mag_filter = 'linear'
        texture.min_filter = 'linear'

        return texture


    def show_compare(self, instance: CompareExample):
        
        if instance.state != 'down' and instance.last_touch.button == 'right':
            self.toolspanel.disabled = False
            snapshots = self.ids.av
            snapshots.remove_widget(instance)
            self.snapshotinfo = ""
            return
        elif instance.state != 'down' and instance.last_touch.button == 'left':
            self.toolspanel.disabled = False
            self.snapshotinfo = ""
            self.trigger_process()            
            return
        
        self.toolspanel.disabled = True

        post, labels = instance.imagedata.get_data('post')

        self.snapshotinfo = '\n'.join(['%s: %s' % (key, value) for (key, value) in labels.items()])
                
        self.imageout = self.get_texture(post)
        self.outparams = labels

        if self.denoise_enabled:                        
            post, labels = self.currentimage.get_data('post')
            self.imageorig = self.get_texture(post)
            self.origparams = labels

    def take_snapshot(self):
        snapshots = self.ids.av  

        if not self.currentimage.has_data('post'):
            return

        compare = CompareExample(
                icon= self.iconpath + '/icons/color-palette-outline.png',
                group='compare',
                imagedata=self.currentimage.get_clone())

        compare.bind(on_release=self.show_compare)
        snapshots.add_widget(compare,3)        
