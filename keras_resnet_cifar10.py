from keras.datasets import cifar10
from keras.models import Model
from keras.utils import np_utils
from keras.layers import Input, Conv2D, Dense
from keras.layers import GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator

def build_plain_architecture():

  inputs = Input(shape=(None,32,32,3))
  
  x = Conv2D(16, kernel_size=(3,3), strides=(2,2), activation='relu')(inputs)
  x = Conv2D(16, kernel_size=(3,3), strides=(2,2), activation='relu')(x)
  x = Conv2D(32, kernel_size=(3,3), strides=(2,2), activation='relu')(x)
  x = Conv2D(32, kernel_size=(3,3), strides=(2,2), activation='relu')(x)
  x = Conv2D(64, kernel_size=(3,3), strides=(2,2), activation='relu')(x)
  x = Conv2D(64, kernel_size=(3,3), strides=(2,2), activation='relu')(x)
  x = GlobalAveragePooling2D()
  predictions = Dense(10, activation='softmax')

  model = Model(inputs=inputs, outputs=predictions)
  model.compile()

def build_residuak_architecture():
  pass

def train_plain_architecture(datagen):
  pass

def train_residual_architecture(datagen):
  pass


def main():
  # constants
  num_classes = 10
  
  # load data set
  (x_train, y_train), (x_test, y_test) = cifar10.load_data()
  y_train = np_utils.to_categorical(y_train, num_classes)
  y_test = np_utils.to_categorical(y_test, num_classes)

  # subtract featurewuse mean as in He et al.2015
  train_datagen = ImageDataGenerator(featurewise_center=True)

  train_datagen.fit(x_train)

  print(x_train.shape)

if __name__ == '__main__':
  main()
