# Convolutional Neural Network
# Dogs vs. Cats

# Part I - Building the Convolutional Neural Network

# Importing the Keras libraries and packages
from keras.models import Sequential
# The convolution step for convolutional layer, 2D because images
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), 
                             activation = 'relu'))
# 32 feature detectors with size 3x3 that will produce 32 feature maps
# inputshape number of channels 3=color images of dimension of 
# input array = 64 by 64 pixels

# Step 2 - Pooling
# consists of reducing size of feature maps
classifier.add(MaxPooling2D(pool_size = (2,2)))
# size 2x2 pool to slide over feature maps

# Step 3 - Flattening
# consists of all pooled feature maps and put them into one single vector
classifier.add(Flatten())

# Step 4 - Full Connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))
 
 # Compiling the CNN
classifier.compile(optimizer = 'adam',
                    loss = 'binary_crossentropy',
                    metrics = ['accuracy'])
 
 # Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

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

classifier.fit_generator(training_set,
                         steps_per_epoch = 80,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 20)

# Part 3 - Making new predictions

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'