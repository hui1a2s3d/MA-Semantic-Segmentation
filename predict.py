'''
predict.py attention:
1、Batch prediction is not possible. If you want batch prediction, use os.listdir() to traverse the folder and use Image.open to open the image file for prediction.
2、Set the blend parameter to False: not mix original image with the segmentation image
4、To get the corresponding area according to the mask: refer to the part of drawing using the prediction result in detect_image.
seg_img = np.zeros((np.shape(pr)[0],np.shape(pr)[1],3))
for c in range(self.num_classes):
    seg_img[:, :, 0] += ((pr == c)*( self.colors[c][0] )).astype('uint8')
    seg_img[:, :, 1] += ((pr == c)*( self.colors[c][1] )).astype('uint8')
    seg_img[:, :, 2] += ((pr == c)*( self.colors[c][2] )).astype('uint8')
'''
from PIL import Image

from pspnet import Pspnet

pspnet = Pspnet()

while True:
    img = input('Input image filename:')
    try:
        image = Image.open(img)
    except:         
        print('Open Error! Try again!')
        continue
    else:
        r_image = pspnet.detect_image(image)
        r_image.show()
        r_image.save("img1.jpg")
