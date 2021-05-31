import cv2
import numpy as np
from threading import Thread
import time

class WebCamera:
    def __init__(self, src, Wd = None, Hd =None, is_unwarp_fisheye = False, image_size = None):


        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        if image_size is not None:
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, image_size[0])
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, image_size[1])

        # self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        # self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()
        if self.frame is None:
            print('%s is not valid!'%src)
            self.stream.release()
            self.stream = None
            return None
        # Hs, Ws = self.frame.shape[:2]
        Ws, Hs = self.size()

        if Wd is None or Hd is None:
            Wd = int(Ws)
            Hd = int(Hs)
        
        self.Wd = Wd
        self.Hd = Hd
        self.is_unwarp = is_unwarp_fisheye
        if is_unwarp_fisheye:
            self.map_x, self.map_y = buildMap(Ws,Hs,Wd,Hd)
        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        if self.stream is None:
            return None
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped or self.stream is None:
                return

            # otherwise, read the next frame from the stream
            
            grabbed, frame = self.stream.read()
            if grabbed:
                self.grabbed, self.frame = grabbed, frame
            else:
                self.grabbed, self.frame = grabbed, None
            time.sleep(0.01)

    def read(self):
        # return the frame most recently read
        frame = self.frame
        if self.is_unwarp:
            frame = unwarp(frame, self.map_x, self.map_y)
        elif frame is not None:
            frame = cv2.resize(frame, (self.Wd, self.Hd))

        return frame

    def size(self):
        # return size of the capture device
        if self.stream is None: return 0,0
        return self.stream.get(3), self.stream.get(4)

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        
    def valid(self):
        return self.grabbed


def buildMap(Ws,Hs,Wd,Hd,method = 0):
    # method can be: Elliptical-Grid, FG-Squircle, Schwarz-Christoffel
    # paul 
    if isinstance(method, int) or method.isnumeric():
        methods = ['Elliptical-Grid', 'FG-Squircle', 'paul', 'Schwarz-Christoffel']
        method = methods[int(method)]

    # Build the fisheye mapping
    map_x = np.zeros((Hd,Wd),np.float32)
    map_y = np.zeros((Hd,Wd),np.float32)

    # Fill in the map, this is slow but
    # we could probably speed it up
    # since we only calc it once, whatever
    for yy in range(0,int(Hd)):
        for xx in range(0,int(Wd)):
            x=  ((xx) / (Wd-1) - 0.5) *2
            y = ((yy) / (Hd-1) - 0.5) *2

            if method == 'Elliptical-Grid':
                xS = (Ws-1) * (0.5 + 0.5* x * np.sqrt(1-y**2/2))
                yS = (Hs-1) * (0.5 +  0.5* y * np.sqrt(1-x**2/2))

            if method == ' FG-Squircle':
                r1 = np.sqrt(x**2 + y**2 - (x**2)*(y**2))
                r2 = np.sqrt(x**2 + y**2)
                r = r1/np.maximum(r2, 0.00000001)

                xS = (Ws-1) * (0.5 + 0.5* x * r)
                yS = (Hs-1) * (0.5 +  0.5* y * r)


            if method == 'paul':
                # Http://http://paulbourke.net/dome/squarefisheye/
                u = np.arcsin(x)
                v = np.arcsin(y)
                z = np.cos(u)*np.cos(v)

                r = np.sqrt(x**2 + y**2 + z**2)
                xS = (Ws-1) * (0.5 + 0.5* x / r)
                yS = (Hs-1) * (0.5 +  0.5* y / r)



            map_x.itemset((yy,xx),xS)
            map_y.itemset((yy,xx),yS)

    return map_x, map_y


def unwarp(img,xmap,ymap):
    if img is None: return None
    # apply the unwarping map to our image
    result = cv2.remap(img,xmap,ymap,cv2.INTER_LINEAR)
    # result = cv2.remap(img,xmap,ymap,cv2.INTER_CUBIC)
    # result = cv2.remap(img,xmap,ymap,cv2.INTER_NEAREST)

    return result
