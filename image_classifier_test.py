import classifier
import numpy as np
from keras.preprocessing import image

image_path = "test/Merle.jpg"
weight = "weights-improvement-25-0.2168.hdf5"

classifier = classifier.create_classifier()
classifier.load_weights(weight)
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


img = image.load_img(image_path, target_size=(64, 64))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)

prediction = classifier.predict(img)
print(prediction)

if prediction[0][0] == 1:
    print('Dog')
else:
    print('Cat')
