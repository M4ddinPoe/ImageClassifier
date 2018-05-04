import classifier
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

image_path = "test/cat.jpg"
weight = "weights-improvement-25-0.2252.hdf5"

classifier = classifier.create_classifier()
classifier.load_weights(weight)
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


img = image.load_img(path=image_path, grayscale=False, target_size=(64, 64, 3))
img = image.img_to_array(img)
test_img = img.reshape((1, 64, 64, 3))

prediction = classifier.predict_classes(test_img, verbose=0)

if prediction == [[0]]:
    print('Cat')
else:
    print('Dog')
