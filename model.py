#importing libraries + data
import tensorflow as tf 
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from google.colab import drive
drive.mount('/content/drive')

#setting up training data
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
training_set = train_datagen.flow_from_directory("/content/drive/MyDrive/kelp/Archive",
    classes=['train'],
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

#setting up test data
test_datagen = ImageDataGenerator(rescale=1. / 255)

test_set = test_datagen.flow_from_directory("/content/drive/MyDrive/kelp/Archive",
    classes=['test'],
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

cnn = tf.keras.models.Sequential()

# convolutional layer:
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(64, 64, 3)))

# pooling layer:
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# one more conv and pooling layer:
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# flattening:
cnn.add(tf.keras.layers.Flatten())
# dense layer:
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# output layer:
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# compiling the CNN:
cnn.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

# training cnn on train data and testing on test data:
cnn.fit(x=training_set, validation_data=test_set, epochs=15, validation_steps=len(test_set))
