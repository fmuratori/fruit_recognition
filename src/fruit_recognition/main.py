import numpy as np

from . import load, opencv, model

from sklearn.model_selection import train_test_split
######################### IMAGE CLASSIFIER ##############################

# get data
X, y, classes_lookup = load.load_apples('./data/fruit_recognition/Apple/', 
                                        opencv.loader)

# preprocess images
X = opencv.process_images(X, grey=False, shape=(150,150))
X = np.stack(X, axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    shuffle=True, random_state=0)

# initialize model
classifier = model.ImageClassifier((150, 150, 3), len(classes_lookup), 
                                   './data/models/apples_classifier_t2.h5')

# train
classifier.fit(X_train, y_train)

# metrics
y_pred = classifier.predict(X_test)
classifier.stats(y_test, y_pred)
classifier.plot_training()
