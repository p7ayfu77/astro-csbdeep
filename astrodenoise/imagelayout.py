from kivy.uix.anchorlayout import AnchorLayout
from kivy.properties import ObjectProperty, NumericProperty
from kivy.graphics.texture import Texture
from kivy.uix.label import Label

class ImageViewLayout(AnchorLayout):
    region_x = NumericProperty(0)
    region_y = NumericProperty(0)
    region_w = NumericProperty(0)
    region_h = NumericProperty(0)
    scale = NumericProperty(1)
    displayimage = ObjectProperty(None)
    labelleft = ObjectProperty(None)
    labelright = ObjectProperty(None)
    imageout = ObjectProperty(None, allownone=True)
    imageorig = ObjectProperty(None, allownone=True)    
    outparams = ObjectProperty(None, allownone=True)
    origparams = ObjectProperty(None, allownone=True)
    
    def setlabels(self):
        self.labelleft.text = '\n'.join(['%s: %s' % (key, value) for (key, value) in self.origparams.items()])
        self.labelright.text = '\n'.join(['%s: %s' % (key, value) for (key, value) in self.outparams.items()])

    def showlabels(self, show: bool = False):
        if show:
            self.labelleft.opacity = 1
            self.labelright.opacity = 1
        else:
            self.labelleft.opacity = 0
            self.labelright.opacity = 0

    def on_imageout(self, instance, value):                
        if value:
            self.region_w = value.width / self.scale
            self.region_h = value.height / self.scale
            self.displayimage.texture = value.get_region(self.region_x, self.region_y, self.region_w, self.region_h)
        else:
            self.region_w = 0
            self.region_h = 0
            self.scale = 1
            self.region_x = 0
            self.region_y = 0

        label = Label()
        label.opacity = 1

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

            self.displayimage.texture = self.imageout.get_region(self.region_x, self.region_y, self.region_w, self.region_h)

        else:
            touch.grab(self)            
      
        if touch.button == 'right' \
            and self.imageorig is not None:
            self.setlabels()
            self.showlabels(True)

        return True

    def on_touch_move(self, touch):
        if not self.collide_point(*touch.pos):
            return super().on_touch_move(touch)

        if self.imageout is None:
            return True

        if touch.grab_current is self \
            and touch.button == 'left':

            imx, imy = self.displayimage.get_norm_image_size()
            deltax = -touch.dx * (self.region_w/imx)
            deltay = -touch.dy * (self.region_h/imy)

            new_x = self.region_x + deltax
            new_y = self.region_y + deltay
            if (new_x >= 0) and (new_x + self.region_w <= self.imageout.width):
                self.region_x += deltax

            if (new_y >= 0) and (new_y + self.region_h <= self.imageout.height):
                self.region_y += deltay

            self.displayimage.texture = self.imageout.get_region(self.region_x, self.region_y, self.region_w, self.region_h)

        #https://stackoverflow.com/questions/74543030/get-location-of-pixel-upon-click-in-kivy
        if touch.grab_current is self \
            and touch.button == 'right' \
            and self.imageorig is not None:
            #touch.sync_with_dispatch = True
            childImageNormImageSize_x = self.displayimage.norm_image_size[0]
            #childImageNormImageSize_y = childImage.norm_image_size[1]
            lr_space = (self.width - childImageNormImageSize_x) / 2  # empty space in Image widget left and right of actual image
            #tb_space = (self.height - childImageNormImageSize_y) / 2  # empty space in Image widget above and below actual image

            pixel_x = touch.x - lr_space - self.x  # x coordinate of touch measured from lower left of actual image
            #pixel_y = touch.y - tb_space - self.y  # y coordinate of touch measured from lower left of actual image
            
            if pixel_x > 0 and pixel_x < childImageNormImageSize_x:
                #clicked inside image, coords: pixel_x, pixel_y

                image_x = int(pixel_x * self.region_w / childImageNormImageSize_x)
                #image_y = pixel_y * self.region_h / childImageNormImageSize_y
                
                if image_x > 0 and image_x < self.region_w:
                    mixtexture = Texture.create(size=(self.region_w, self.region_h), colorfmt=self.imageout.colorfmt)
                    
                    mixtexture.blit_buffer(
                        self.imageout.get_region(self.region_x + image_x, self.region_y, self.region_w - image_x, self.region_h).pixels,
                        pos=(image_x, 0), 
                        size=(self.region_w - image_x, self.region_h),
                        bufferfmt='ubyte',
                        colorfmt='rgba')

                    mixtexture.blit_buffer(
                        self.imageorig.get_region(self.region_x, self.region_y, image_x, self.region_h).pixels,
                        pos=(0,0), 
                        size=(image_x, self.region_h),
                        bufferfmt='ubyte',
                        colorfmt='rgba')                                
                    
                    mixtexture.flip_vertical()
                    mixtexture.mag_filter = 'linear'
                    mixtexture.min_filter = 'linear'
                    self.displayimage.texture = mixtexture

        return True

    def on_touch_up(self, touch):
        super().on_touch_up(touch)
        
        if touch.grab_current is self:
            touch.ungrab(self)

        self.showlabels(False)

        if self.imageout is not None:
            self.displayimage.texture = self.imageout.get_region(self.region_x, self.region_y, self.region_w, self.region_h)
        
        return True
