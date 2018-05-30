from keras.datasets import cifar10
from keras.models import Model
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.layers import Input, Conv2D, Dense
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

def build_plain_architecture():
  '''
    Architecture of ResNET (He et al. 2015) with n=1.
  '''

  inputs = Input(shape=(32,32,3))
  
  x = Conv2D(16, kernel_size=(3,3), strides=(2,2), activation='relu', \
    kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(inputs)
  x = BatchNormalization()(x)
  x = Conv2D(16, kernel_size=(3,3), strides=(1,1), activation='relu', \
    kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
  x = BatchNormalization()(x)
  x = Conv2D(16, kernel_size=(3,3), strides=(1,1), activation='relu', \
    kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
  x = BatchNormalization()(x)
  x = Conv2D(32, kernel_size=(3,3), strides=(1,1), activation='relu', \
    kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
  x = BatchNormalization()(x)
  x = Conv2D(32, kernel_size=(3,3), strides=(1,1), activation='relu', \
    kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
  x = BatchNormalization()(x)
  x = Conv2D(64, kernel_size=(3,3), strides=(1,1), activation='relu', \
    kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
  x = BatchNormalization()(x)
  x = Conv2D(64, kernel_size=(3,3), strides=(1,1), activation='relu', \
    kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
  x = GlobalAveragePooling2D()(x)
  x = BatchNormalization()(x)
  predictions = Dense(10, activation='softmax', \
    kernel_initializer='he_normal')(x)

  model = Model(inputs=inputs, outputs=predictions)
  model.compile(optimizer=SGD(lr=0.1, decay=0.0001, momentum=0.9), \
    loss='categorical_crossentropy')

  model.summary()

  return model

def build_residual_architecture():
  pass

def train_plain_architecture(x, y, x_valid, y_valid, datagen, valid_datagen):

  model = build_plain_architecture()

  model.fit_generator(datagen.flow(x, y, batch_size=128), \
    validation_data=valid_datagen.flow(x_valid, y_valid), \
    validation_steps=x_valid.shape[0]/128, \
    steps_per_epoch=x.shape[0]/128, epochs=32000)

  # after 32k iterations divide learning rate by 10
  model.compile(optimizer=SGD(lr=0.01, decay=0.0001, momentum=0.9), \
    loss='categorical_crossentropy')
    
  model.fit_generator(datagen.flow(x, y, batch_size=128), \
    validation_data=valid_datagen.flow(x_valid, y_valid), \
    validation_steps=x_valid.shape[0]/128, \
    steps_per_epoch=x.shape[0]/128, epochs=16000)

  # after 48k iterations divide learning rate by 10
  model.compile(optimizer=SGD(lr=0.001, decay=0.0001, momentum=0.9), \
    loss='categorical_crossentropy')
    
  model.fit_generator(datagen.flow(x, y, batch_size=128), \
    validation_data=valid_datagen.flow(x_valid, y_valid), \
    validation_steps=x_valid.shape[0]/128, \
    steps_per_epoch=x.shape[0]/128, epochs=16000)

  model.save('keras_plain_cifar10.h5')
  return model

def train_residual_architecture(datagen):
  pass


def main():
  # constants
  num_classes = 10
  
  # load data set
  (x_train, y_train), (x_test, y_test) = cifar10.load_data()
  x_train = x_train / 255.
  x_valid = x_train[45000:]
  y_valid = y_train[45000:]
  x_train = x_train[:45000]
  y_train = y_train[:45000]
  x_test = x_test / 255.
  y_train = np_utils.to_categorical(y_train, num_classes)
  y_valid = np_utils.to_categorical(y_valid, num_classes)
  y_test = np_utils.to_categorical(y_test, num_classes)

  # subtract featurewise mean, horizontal flip and padded cropping as in He et al.2015
  train_datagen = ImageDataGenerator(featurewise_center=True, \
                                     horizontal_flip=True, \
                                     width_shift_range=4, \
                                     height_shift_range=4)

  train_datagen.fit(x_train)

  print(x_train.shape)

  valid_datagen = ImageDataGenerator(featurewise_center=True, \
                                     horizontal_flip=True, \
                                     width_shift_range=4, \
                                     height_shift_range=4)

  valid_datagen.fit(x_valid)

  model = train_plain_architecture(x_train, y_train, x_valid, y_valid, train_datagen, valid_datagen)

if __name__ == '__main__':
  main()
