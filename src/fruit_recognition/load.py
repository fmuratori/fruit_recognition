import os
import json
import imghdr
import logging
import pandas as pd

log = logging.getLogger(__name__)

############################## CUSTOM LOAD ################################
APPLE_DATASET = {"Apple": ["Apple A", "Apple B", "Apple C", "Apple D", "Apple E", "Apple F"]}

FRUIT_DATASET_v1 = [
    {"Apple": ["Total number of Apples"]},
    "Banana",
    "Carambola",
    {"Guava": ["guava total final"]},
    {"Kiwi": ["Total number of Kiwi fruit"]},
    "Mango",
    "muskmelon", 
    "Orange",
    "Peach",
    "Pear",
    "Persimmon",
    "Pitaya",
    "Plum",
    "Pomegranate",
    "Tomatoes"]

FRUIT_DATASET_v2 = [
    {"Apple": ["Apple A", "Apple B", "Apple C", "Apple D", "Apple E", "Apple F"]},
    "Banana",
    "Carambola",
    {"Guava": ["guava A", "guava B"]},
    {"Kiwi": ["kiwi A", "Kiwi B", "Kiwi C"]},
    "Mango",
    "muskmelon", 
    "Orange",
    "Peach",
    "Pear",
    "Persimmon",
    "Pitaya",
    "Plum",
    "Pomegranate",
    "Tomatoes"]

def load_fruits(path, loader, structure):
    if not os.path.isdir(path):
        raise ValueError('Path {} is not a directory')

    images = []
    for ffolder in structure:
        if isinstance(ffolder, str): # folder without child folders
            images.extend(load_folder_images(path, ffolder, loader))
        elif isinstace(ffolder, dict): # folder with child folders
            (class_name, child_classes) = ffolder.items()
            child_path = os.path.join(path, class_name)
            load_fruits(child_path, loader, structure)

    return images

############################## LOAD UTILS ##############################

def is_valid_image(path, img_type=None):
    '''
    Check if a given path is an image type. If img_type is not None, the image 
    type must also match the specified string value.
    '''
    try:
      real_img_type = imghdr.what(path)
      return real_img_type is not None and (real_img_type == img_type or img_type is None)
    except:
      return False

def load_folder_images(path, class_name, loader):
    path = os.path.join(path, class_name)
    if not os.path.isdir(path):
        raise ValueError('Specified path {} is not a folder'.format(path))
    images = [(loader(os.path.join(folder,filename)), class_name) 
              for filename in sorted(os.listdir(path)) 
              if is_valid_image(filepath)]
    log.debug('Loaded {} images of class {}'.format(len(images), class_name))
    return images


if __name__ == '__main__':
    pass