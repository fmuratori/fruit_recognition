import numpy as np
import logging
import click
import time
import os

from . import load, opencv, model

from sklearn.model_selection import train_test_split

# logging utility
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(message)s', 
                    datefmt='%m/%d/%Y %H:%M:%S')
log = logging.getLogger(__name__)

######################### MULTI CLASS IMAGE CLASSIFIER ########################

@click.command()
@click.option("--target", default='apples', help="Select target of the classification routine. values: apple or fruit")
def fruit_classifier(target='apple'):
    log.info('Starting fruit classification ...')   

    # get data
    log.info('Loading images...')
    start_time = time.time()
    if target == 'apple':
        target = load.APPLE_DATASET
    elif target == 'fruit1':
        target = load.FRUIT_DATASET_v1
    elif target == 'fruit2':
        target = load.FRUIT_DATASET_v2
    images = load.load_(images, opencv.loader, target)
    end_time = time.time()
    log.info('Loaded {} images in {} seconds'.format(len(images), 
                                                     int(end_time-start_time)))

    # preprocess images
    log.info('Preprocessing data ...')
    start_time = time.time()
    images = pd.DataFrame(images, columns=['image', 'fruit'])
    classes_set = set(images.fruit)
    classes_lookup = dict(zip(sorted(classes_set), range(len(classes_set))))
    y = images.fruit.apply(lambda y: classes_lookup[y]).values
    X = images.image.values
    X = opencv.process_images(X, grey=False, shape=(200,200))
    X = np.stack(X, axis=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
        shuffle=True, random_state=0)
    end_time = time.time()
    log.info('Loaded images in {} seconds'.format(int(end_time-start_time)))

    # model train
    log.info('Training classifier model ...')
    start_time = time.time()
    classifier = model.ImageClassifier((200, 200, 3), len(classes_lookup), 
        os.path.join(data,  'models', 'apples_classifier_t2.h5'))
    classifier.fit(X_train, y_train)
    end_time = time.time()
    log.info('Classifier model trained in {} seconds'.format(int(end_time - start_time)))

    # metrics
    log.info('=== Metrics ===')
    y_pred = classifier.predict(X_test)
    classifier.stats(y_test, y_pred)
    classifier.plot_training()

############################## MULTI LABEL CLASSIFIER #########################


if __name__ == "__main__":
    fruit_classifier()