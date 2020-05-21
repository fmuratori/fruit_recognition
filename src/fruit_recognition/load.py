import os
import imghdr

import pandas as pd


############################## HIGH LEVEL LOAD ################################

def load_apples(folder, loader):
    apples = []

    for foldername in sorted(os.listdir(folder)):
        path = os.path.join(folder, foldername)
        if os.path.isdir(path) and foldername != 'Total Number of Apples':
            apples.extend([(img, foldername) 
                            for img in get_images_from_folder(path, loader)])
    
    apples = pd.DataFrame(apples, columns=['image', 'fruit'])

    classes_set = set(apples.fruit)
    classes_lookup = dict(zip(sorted(classes_set), range(len(classes_set))))

    y = apples.fruit.apply(lambda y: classes_lookup[y]).values
    X = apples.image.values

    return X, y, classes_lookup


############################## LOAD UTILS ##############################

def is_valid_image(path, img_type=None):
    '''
    Check if a given path is an image type. If img_type is not None, the image 
    type must also match the specified string value.
    '''
    real_img_type = imghdr.what(path)
    return real_img_type is not None and (real_img_type == img_type or img_type is None)

def get_images_from_folder(folder, image_loader):
    images = []
    for filename in sorted(os.listdir(folder)):
        path = os.path.join(folder,filename)
        if is_valid_image(path):
            images.append(image_loader(path))
    return images

if __name__ == '__main__':
    pass
    # test
    # print(get_images_from_folder('./data/sample/', opencv.loader))
    # print(load_apples('./data/sample/Apple'))