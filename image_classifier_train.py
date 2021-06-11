import classifier
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

classifier = classifier.create_classifier()

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

checkpoint = ModelCheckpoint("weights-improvement-{epoch:02d}-{loss:.4f}.hdf5", monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit(x = training_set,
               validation_data=test_set,
               epochs = 25,
               callbacks=callbacks_list)