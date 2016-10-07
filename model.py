'''
Construct different DNN models
'''

from keras.applications import vgg16
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Dropout, Input, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras import backend as K

def build_vgg16(image_size=None):
	image_size = image_size or (240, 240)
	if K.image_dim_ordering() == 'th':
	    input_shape = (3,) + image_size
	else:
	    input_shape = image_size + (3, )
	bottleneck_model = vgg16.VGG16(include_top=False, 
	                               input_tensor=Input(input_shape))
	#bottleneck_model.trainable = False
	for layer in bottleneck_model.layers:
	    layer.trainable = False

	x = bottleneck_model.input
	y = bottleneck_model.output
	y = Flatten()(y)
	y = BatchNormalization()(y)
	y = Dense(2048, activation='relu')(y)
	y = Dropout(.5)(y)
	y = Dense(1024, activation='relu')(y)
	y = Dropout(.5)(y)
	y = Dense(1)(y)

	model = Model(input=x, output=y)
	model.compile(optimizer=Adam(lr=1e-4), loss = 'mse')
	return model

def build_cnn(image_size=None):
	image_size = image_size or (60, 80)
	if K.image_dim_ordering() == 'th':
	    input_shape = (3,) + image_size
	else:
	    input_shape = image_size + (3, )

	img_input = Input(input_shape)

	x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(img_input)
	x = Dropout(0.5)(x)
	x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(x)
	x = Dropout(0.5)(x)
	x = MaxPooling2D((2, 2), strides=(2, 2))(x)

	x = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(x)
	x = Dropout(0.5)(x)
	# it doesn't fit in my GPU
	# x = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(x)
	# x = Dropout(0.5)(x)
	x = MaxPooling2D((2, 2), strides=(2, 2))(x)

	y = Flatten()(x)
	y = Dense(1024, activation='relu')(y)
	y = Dropout(.5)(y)
	y = Dense(1024, activation='relu')(y)
	y = Dropout(.5)(y)
	y = Dense(1)(y)

	model = Model(input=img_input, output=y)
	model.compile(optimizer=Adam(lr=1e-4), loss = 'mse')
	return model