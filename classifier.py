from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense

def create_classifier():
    classifier = Sequential()

    classifier.add(Conv2D(filters=32, kernel_size=3, input_shape = ( 64, 64, 3), activation = 'relu'))
    classifier.add(MaxPool2D(pool_size=2, strides=2))

    classifier.add(Conv2D(filters=32, kernel_size=3, activation = 'relu'))
    classifier.add(MaxPool2D(pool_size=2, strides=2))

    classifier.add(Flatten())
    classifier.add(Dense(units = 128, activation = 'relu'))
    classifier.add(Dense(units = 1, activation = 'sigmoid'))

    return classifier