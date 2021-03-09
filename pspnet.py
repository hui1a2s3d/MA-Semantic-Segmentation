import colorsys
import copy
import os

import numpy as np
from PIL import Image

from nets.pspnet import pspnet


def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image,nw,nh

#--------------------------------------------#
#   3 parameters need to be modified to predict using the self-trained model
#   model_path、backbone and num_classes need to be modified！
#   If there is a shape mismatch, attention to the modified parameters
#--------------------------------------------#
class Pspnet(object):
    _defaults = {
        "model_path"        : 'logs\ep095-loss0.816-val_loss0.964.h5',
        "backbone"          : "mobilenet",
        "model_image_size"  : (473, 473, 3),
        "num_classes"       : 2,
        "downsample_factor" : 16,
        #--------------------------------#
        #   blend: to control whether the recognition result is mixed with the original image
        #--------------------------------#
        "blend"             : True,
        #---------------------------------------------------------------------#
        #   This variable is used to control whether to use letterbox_image to resize the input image without distortion,
        #   Try out True or False, sometimes it has a positive effect, and sometimes it has a negative effect.
        #   The default setting is the better setting method in the pre-training data set.
        #---------------------------------------------------------------------#
        "letterbox_image"   : False,
    }

    #---------------------------------------------------#
    #   Initialization PSPNET
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.generate()

    #---------------------------------------------------#
    #   Load model
    #---------------------------------------------------#
    def generate(self):
        #-------------------------------#
        #   Load model and weights
        #-------------------------------#
        self.model = pspnet(self.num_classes,self.model_image_size,
                    downsample_factor=self.downsample_factor, backbone=self.backbone, aux_branch=False)
        self.model.load_weights(self.model_path, by_name=True)
        print('{} model loaded.'.format(self.model_path))
        
        if self.num_classes <= 21:
            self.colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), 
                    (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), 
                    (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), (128, 64, 12)]
        else:
            # Set different colors for the picture frame
            hsv_tuples = [(x / self.num_classes, 1., 1.)
                        for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(
                map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                    self.colors))


    #---------------------------------------------------#
    #   detect image
    #---------------------------------------------------#
    def detect_image(self, image):
        #---------------------------------------------------#
        #   Backup  the input image and use it for drawing later
        #---------------------------------------------------#
        old_img = copy.deepcopy(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]
        
        #---------------------------------------------------------#
        #   Add gray bars to the image to achieve undistorted resize
        #   can also directly resize
        #---------------------------------------------------------#
        if self.letterbox_image:
            img, nw, nh = letterbox_image(image,(self.model_image_size[1],self.model_image_size[0]))
        else:
            img = image.convert('RGB')
            img = img.resize((self.model_image_size[1],self.model_image_size[0]), Image.BICUBIC)
        img = np.asarray([np.array(img)/255])
        
        #---------------------------------------------------#
        #   Images are transmitted to the network for prediction
        #---------------------------------------------------#
        pr = self.model.predict(img)[0]
        #---------------------------------------------------#
        #   Get the class of each pixel
        #---------------------------------------------------#
        pr = pr.argmax(axis=-1).reshape([self.model_image_size[0], self.model_image_size[1]])
        #--------------------------------------#
        #   Cut off the gray bar
        #--------------------------------------#
        if self.letterbox_image:
            pr = pr[int((self.model_image_size[0]-nh)//2):int((self.model_image_size[0]-nh)//2+nh), int((self.model_image_size[1]-nw)//2):int((self.model_image_size[1]-nw)//2+nw)]
        
        #------------------------------------------------#
        #   Create a new image and assign colors according to the type of each pixel
        #------------------------------------------------#
        seg_img = np.zeros((np.shape(pr)[0],np.shape(pr)[1],3))
        for c in range(self.num_classes):
            seg_img[:, :, 0] += ((pr == c)*( self.colors[c][0] )).astype('uint8')
            seg_img[:, :, 1] += ((pr == c)*( self.colors[c][1] )).astype('uint8')
            seg_img[:, :, 2] += ((pr == c)*( self.colors[c][2] )).astype('uint8')

        #------------------------------------------------#
        #   Convert the new picture into Image form
        #------------------------------------------------#
        image = Image.fromarray(np.uint8(seg_img)).resize((orininal_w,orininal_h), Image.NEAREST)

        #------------------------------------------------#
        #   Mix the new picture with the original picture
        #------------------------------------------------#
        if self.blend:
            image = Image.blend(old_img,image,0.7)

        return image
