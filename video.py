#-------------------------------------#
#   Call camera or video for detection
#   Call the camera and run it directly
#   To call the video, specify the path of cv2.VideoCapture()
#-------------------------------------#
import time

import cv2
import numpy as np
from PIL import Image

from pspnet import Pspnet

pspnet = Pspnet()
#-------------------------------------#
#   Call camera:
#   capture=cv2.VideoCapture("1.mp4")
#-------------------------------------#
capture=cv2.VideoCapture(0)

fps = 0.0
while(True):
    t1 = time.time()
    # Read a frame
    ref,frame=capture.read()
    # BGRtoRGB
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    # convert to Image
    frame = Image.fromarray(np.uint8(frame))
    # Detection
    frame = np.array(pspnet.detect_image(frame))
    # RGBtoBGR meet the opencv format
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

    fps  = ( fps + (1./(time.time()-t1)) ) / 2
    print("fps= %.2f"%(fps))
    frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("video",frame)

    c= cv2.waitKey(30) & 0xff 
    if c==27:
        capture.release()
        break
