import argparse
import csv
import cv2
import math
import numpy as np
import os
import sklearn
import sklearn.model_selection

if not __name__ == '__main__':
    sys.exit(1)

parser = argparse.ArgumentParser()
parser.add_argument(
    'models_dir',
    type = str,
    help = 'Dir in which to save a model per epoch. Must exist and be empty.'
)
args = parser.parse_args()

assert os.listdir(args.models_dir) == [], 'Dir in which to save models is not empty. Aborting.'

# `logs` will be a 2-dim array, with each row containing
# [dataset dir string, array containing data from one log line]
logs = []
for dataset_dir in os.listdir('datasets'):
    dataset_dir = 'datasets/{}'.format(dataset_dir)

    with open('{}/driving_log.csv'.format(dataset_dir)) as f:
        reader = csv.reader(f)
        
        first_line = next(reader, None)
        if not first_line[:2] == ['center', 'left']:
            logs.append([dataset_dir, first_line])
        
        for line in reader:
            logs.append([dataset_dir, line])

print('Num of logs is ', len(logs), '. Compare this with `wc -l datasets/*/*.csv`.')

# in `generator` below, from each log, this many samples are extracted.
generator_multiplicity = 6

def generator(logs, batch_size = math.floor(128 / generator_multiplicity)):
    num_logs = len(logs)
    
    img_path = lambda dataset_dir, orig_file_path: '{}/IMG/{}'.format(dataset_dir, orig_file_path.split('/')[-1])
    
    steering_correction = 0.2

    while 1:
        logs = sklearn.utils.shuffle(logs)
        
        for offset in range(0, num_logs, batch_size):
            log_batch = logs[offset:offset+batch_size]

            images = []
            angles = []
            for dataset_dir, log in log_batch:
                center,left,right,steering,throttle,brake,speed = log
                
                center_image = cv2.imread(img_path(dataset_dir, center))
                center_angle = float(steering)
                
                left_image = cv2.imread(img_path(dataset_dir, left))
                left_angle = center_angle + steering_correction

                right_image = cv2.imread(img_path(dataset_dir, right))
                right_angle = center_angle - steering_correction

                for img in [center_image, left_image, right_image]:
                    images.extend([img, np.fliplr(img)])
                for ang in [center_angle, left_angle, right_angle]:
                    angles.extend([ang, -ang])

            X_train = np.array(images)
            y_train = np.array(angles)
            yield X_train, y_train

train_logs, validation_logs = sklearn.model_selection.train_test_split(logs, test_size = 0.2)

train_generator = generator(train_logs)
validation_generator = generator(validation_logs)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
# from keras.layers.pooling import MaxPooling2D
# from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

model = Sequential()
model.add(Cropping2D(cropping=((70,20),(0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# adam = Adam(lr=0.01, decay=0.005)
model.compile(loss='mse', optimizer='adam')

checkpoint = ModelCheckpoint(args.models_dir + '/model_{epoch:02d}.h5', verbose=1)
history_object = model.fit_generator(
    train_generator,
    samples_per_epoch = len(train_logs) * generator_multiplicity,
    validation_data = validation_generator,
    nb_val_samples = len(validation_logs) * generator_multiplicity, 
    nb_epoch = 10,
    callbacks = [checkpoint],
    verbose = 1
)

import matplotlib
matplotlib.use('SVG')
import matplotlib.pyplot as plt

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('{}/history.png'.format(args.models_dir))
