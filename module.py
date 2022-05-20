import cv2
import numpy as np


class TextSceneDetection(object):
    def __init__(self, model_path):
        self.net = cv2.dnn.readNet(model_path)
        self.input_shape = (512, 512)
    
    def sigmoid_array(self, x):                                        
        return 1 / (1 + np.exp(-x))
    
    def display(self, contours, image):
        for cnt in contours:
            x, y, w, h = cnt
            cv2.rectangle(image,(x,y),(x + w + 10,y + h + 10), (0,255,0), 2)
        return image
    
    def processing(self, mask, h, w):
        out = np.transpose(np.squeeze(mask, 0), (1,2,0))
        out = self.sigmoid_array(out)
        out = cv2.resize(out, (h, w))
        out[out > 0.7] = 255
        out[out <= 0.7] = 0
        kernel = np.ones((3,3),np.uint8)
        dilated = cv2.dilate(out.astype("uint8"), kernel, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cv2.boundingRect(cnt) for cnt in contours]
        return contours
    
    def __call__(self, image):
        image = cv2.imread(image)
        h, w = image.shape[:2]
        img_blob = cv2.dnn.blobFromImage(image, 1/255., self.input_shape, swapRB = True, crop=False)
        self.net.setInput(img_blob)
        mask = self.net.forward()
        contours = self.processing(mask, w, h)
        image_data = self.display(contours, image)
        return image_data[:,:,::-1]
