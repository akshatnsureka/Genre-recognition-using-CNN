from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Input
from keras.layers import GRU
from keras.layers import concatenate
from keras.models import Model
from keras.applications.vgg16 import VGG16
from song_analysis import mfccs

DROPOUT_RATE = 0.5
CONV_DROPOUT_RATE = 0.2


def create_model(input_shape, nb_genre):

    model = Sequential()

    model.add(Conv1D(filters=16,
                     kernel_size=3,
                     input_shape=input_shape,
                     activation='relu',
                     padding='same'))
    model.add(Conv1D(filters=16,
                     kernel_size=3,
                     activation='relu',
                     padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(CONV_DROPOUT_RATE))

    model.add(Conv1D(filters=32,
                     kernel_size=3,
                     activation='relu',
                     padding='same'))
    model.add(Conv1D(filters=32,
                     kernel_size=3,
                     activation='relu',
                     padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(CONV_DROPOUT_RATE))

    model.add(Conv1D(filters=64,
                     kernel_size=3,
                     activation='relu',
                     padding='same'))
    model.add(Conv1D(filters=64,
                     kernel_size=3,
                     activation='relu',
                     padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(CONV_DROPOUT_RATE))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(DROPOUT_RATE))

    model.add(Dense(nb_genre, activation='softmax'))

    return model

def feature_mfcc():
    for i in range(1, 21):
        header += f' mfcc{i}'
        header += ' label'
        header = header.split()
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        to_append += f' {g}'
        file = open('data.csv', 'a', newline='')

def create_rcnn(input_shape, nb_genre):

    input = Input(shape=input_shape)

    convolution = Conv1D(filters=16, kernel_size=3, activation='relu', padding='same')(input)
    convolution = Conv1D(filters=16, kernel_size=3, activation='relu', padding='same')(convolution)
    convolution = MaxPooling1D(pool_size=2)(convolution)
    convolution = Dropout(CONV_DROPOUT_RATE)(convolution)
    convolution = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(convolution)
    convolution = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(convolution)
    convolution = MaxPooling1D(pool_size=2)(convolution)
    convolution = Dropout(CONV_DROPOUT_RATE)(convolution)
    convolution = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(convolution)
    convolution = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(convolution)
    convolution = MaxPooling1D(pool_size=2)(convolution)
    convolution = Dropout(CONV_DROPOUT_RATE)(convolution)
    convolution = Flatten()(convolution)

    recurrent = MaxPooling1D(pool_size=2)(input)
    recurrent = GRU(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(recurrent)
    recurrent = GRU(32, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(recurrent)
    recurrent = GRU(16, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)(recurrent)

    merged = concatenate([convolution, recurrent], axis=1)

    out = Dense(512, activation='relu')(merged)
    out = Dense(512, activation='relu')(out)
    out = Dense(nb_genre, activation='softmax')(out)

    model = Model(input, out)

    return model


def create_model_vgg(input_shape, nb_genre):

    vgg16 = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    output_shape = vgg16.output_shape
    output = vgg16.output

    top = Sequential()
    top.add(Flatten(input_shape=output_shape[1:]))
    top.add(Dense(256, activation='relu'))
    top.add(Dropout(DROPOUT_RATE))
    top.add(Dense(nb_genre, activation='softmax'))

    model = Model(inputs=vgg16.input, outputs=top(output))

    for layer in model.layers[:5]:
        layer.trainable = False

    return model


def create_model_old(input_shape, nb_genre):

    model = Sequential()
    model.add(
        Conv2D(filters=16, kernel_size=(3, 3), input_shape=input_shape, activation='relu'))
    model.add(Conv2D(filters=16, kernel_size=(3, 3), input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(CONV_DROPOUT_RATE))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=input_shape, activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(CONV_DROPOUT_RATE))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=input_shape, activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(CONV_DROPOUT_RATE))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(nb_genre, activation='softmax'))

    return model

