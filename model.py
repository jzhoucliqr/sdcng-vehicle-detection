import numpy as np
import csv
import cv2
import os.path
import matplotlib.image as mpimg

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

lines = []
with open('./all.csv') as f:
    reader = csv.reader(f)
    for line in reader:
        lines.append(line)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            labels = []
            for line in batch_samples:
                image = mpimg.imread(line[0])
                if (line[0].endswith(".jpg")):
                    image = image.astype(np.float32)/255
                label = int(line[1])
                images.append(image)
                labels.append(label)

            X_train = np.array(images)
            y_train = np.array(labels)

            image_flipped = np.fliplr(X_train)
            X_train = np.append(X_train, image_flipped, axis=0)
            y_train = np.append(y_train, labels, axis=0)

            yield shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

model = Sequential()
model.add(Lambda(lambda x : x - 0.5, input_shape=(64,64,3)))
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit_generator(train_generator, samples_per_epoch=2*len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=2)

model.save('model.h5')

