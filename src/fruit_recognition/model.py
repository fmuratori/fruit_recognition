import matplotlib.pyplot as plt
import pandas as pd

from tensorflow import keras
from keras import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class ImageClassifier():
    def __init__(self, img_shape, num_classes, save_path):

        self.model = Sequential()

        self.model.add(Conv2D(filters=32, kernel_size=(3,3),input_shape=img_shape, 
                              activation='relu', padding = 'same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', 
                              padding = 'same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', 
                              padding = 'same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', 
                              padding = 'same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', 
                              padding = 'same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', 
                              padding = 'same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())

        self.model.add(Dense(256))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))

        self.model.add(Dense(num_classes))
        self.model.add(Activation('softmax'))

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', 
                           metrics=['accuracy'])

        self.callbacks = [EarlyStopping(monitor='val_loss', patience=10),
                          ModelCheckpoint(filepath=save_path, 
                                          monitor='val_loss', save_best_only=True)]

    def fit(self, X, y):
        self.model.fit(X, to_categorical(y), batch_size=128, epochs=100, 
                       callbacks=self.callbacks, validation_split=0.1, 
                       verbose=1)

    def predict(self, X):
        return self.model.predict_classes(X)

    def stats(self, y_test, y_pred):
        # Display prediction statistics

        print(f"### Result of the predictions using {len(y_test)} test data ###\n")
        print("Classification Report:\n")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:\n\n")
        print(confusion_matrix(y_test, y_pred))
        print("\nAccuracy:", round(accuracy_score(y_test, y_pred),5))
        
    def plot_training(self):
        history = pd.DataFrame(self.model.history.history)
        history[["accuracy","val_accuracy"]].plot()
        plt.title("Training results")
        plt.xlabel("# epoch")
        plt.show()

    def _from_categorical(self, ohe_list):
        """
        Inverse of to_categorical
        Example: [[0,0,0,1,0], [1,0,0,0,0]] => [3,0]
        """
        ohe_list = ohe_list.tolist()
        output = []
        for x in ohe_list:
            output.append(x.index(max(x)))
        return output