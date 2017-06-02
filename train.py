'''
This is our directory structure:
```
data/
    train/
        class1/
            c1t001.jpg
            c1t002.jpg
            ...
        class2/
            c2t001.jpg
            c2t002.jpg
            ...
    validation/
        class1/
            c1v001.jpg
            c1v002.jpg
            ...
        class2/
            c2v001.jpg
            c2v002.jpg
            ...
```
'''
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications

# dimensions of our images.
img_width, img_height = 150, 150

top_model_path = 'bottleneck_fc_model.h5'
top_weights_path = 'bottleneck_fc_weights.h5'
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 50
batch_size = 16


def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')
    print('using VGG16 model')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    print('generator')

    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    print('before saving bottleneck train')

    np.save(open('bottleneck_features_train.npy', 'w'),
            bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    print('generator')

    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)

    print('before save to validation')
    np.save(open('bottleneck_features_validation.npy', 'w'),
            bottleneck_features_validation)


def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy'))
    print('loaded bottleneck train data')

    train_labels = np.array(
        [0] * (nb_train_samples / 2) + [1] * (nb_train_samples / 2))
    print('created train labels')

    validation_data = np.load(open('bottleneck_features_validation.npy'))
    validation_labels = np.array(
        [0] * (nb_validation_samples / 2) + [1] * (nb_validation_samples / 2))
    print('created validation data and label')

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])
    print('model compiled')

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    print('model fitted')

    model.save(top_model_path)
    model.save_weights(top_weights_path)
    print("saved model and weights")


save_bottlebeck_features()
train_top_model()


