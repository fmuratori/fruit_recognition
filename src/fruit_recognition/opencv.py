import numpy as np
import cv2


####################### OPENCV HIGH LEVEL IMAGE PROCESSING ##########################

def process_images(images, grey=False, shape=None):
    '''
    Apply generic image processing functions to each provided image:
    - greyscale: apply color transformation from rgb to grayscale
    - shape: resize images to specified pixels dimensions. If not None, a tuple 
    of (width, height) values must be set.
    '''

    if shape is not None:
        images = [cv2.resize(img, shape) for img in images]

    if grey:
        images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]

    return images


########################## OPENCV FUNCTIONS WRAPPER ############################

def loader(path):
    return np.array(cv2.imread(path), dtype=np.uint8)
