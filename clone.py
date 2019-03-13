import csv
import cv2
import numpy as np

lines = []
with open('/opt/carnd_p3/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    firstLineRead = False
    for line in reader:
        if not firstLineRead:
            firstLineRead = True
        else:
            lines.append(line)
        
images = []
measurements = []
correction = 0.2
steering_correction=[0.0, correction, -correction]
for line in lines:
    for i in range(3):
        source_path = line[i]
        filename=source_path.split('/')[-1]
        #print(filename)
        current_path = '/opt/carnd_p3/data/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
        images.append(cv2.flip(image,1))
        measurement = float(line[3])+steering_correction[i]
        measurements.append(measurement)
        measurements.append(measurement*-1.0)

x_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Dropout

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160,320,3)))
model.add(Cropping2D(cropping=((70,0),(0,0))))
from keras.applications.inception_v3 import InceptionV3

model.add(InceptionV3(weights='imagenet', include_top=False))
#model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss = 'mse' , optimizer = 'adam')
history_object=model.fit(x_train, y_train, validation_split=0.2, shuffle = True, epochs = 2)
#model.fit(x_train, y_train, epochs = 7)
model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())
import matplotlib.pyplot as plt
### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()