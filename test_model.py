'''
Test different models on udacity self-driving car dataset

'''
from __future__ import print_function

# for docker env
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import argparse
import numpy as np
from os import path
import time

from data import *
from model import *

from keras.callbacks import ModelCheckpoint

def main():
	# parse arguments
	parser = argparse.ArgumentParser(description="Testing Udacity SDC data")
	parser.add_argument('--dataset', type=str, help='dataset folder with csv and image folders')
	parser.add_argument('--model', type=str, help='model to evaluate, current list: {vgg16, cnn}')
	parser.add_argument('--resized-image-width', type=int, help='image resizing')
	parser.add_argument('--resized-image-height', type=int, help='image resizing')
	parser.add_argument('--nb-epoch', type=int, help='# of training epoch')
	parser.add_argument('--camera', type=str, default='center', help='camera to use, default is center')
	parser.add_argument('--batch_size', type=int, default=8, help='training batch size')
	args = parser.parse_args()

	dataset_path = args.dataset
	model_name = args.model
	image_size = (args.resized_image_width, args.resized_image_height)
	camera = args.camera
	batch_size = args.batch_size
	nb_epoch = args.nb_epoch

	# build model and train it
	steering_log = path.join(dataset_path, 'steering.csv')
	image_log = path.join(dataset_path, 'camera.csv')
	camera_images = dataset_path

	model_builders = {
			'vgg16': (build_vgg16, vgg_preprocess_input, exact_output) 
			, 'cnn': (build_cnn, normalize_input, exact_output)
	}

	if model_name not in model_builders:
		raise ValueError("unsupported model %s" % model_name)
	model_builder, input_processor, output_processor = model_builders[model_name]
	model = model_builder(image_size)
	print('model %s built...' % model_name)

	train_generator = data_generator(steering_log=steering_log,
                                 image_log=image_log, 
                                 image_folder=camera_images,
                                 camera=camera,
                                 batch_size=batch_size,
                                 image_size=image_size,
                                 timestamp_end=14751877262-300, # exclude last 30s in training
                                 shuffle=True,
                                 preprocess_input=input_processor,
                                 preprocess_output=output_processor)
	model_saver = ModelCheckpoint(filepath="%s_weights.hdf5" % model_name, verbose=1, save_best_only=False)
	model.fit_generator(train_generator, samples_per_epoch=35000, nb_epoch=nb_epoch,
						callbacks=[model_saver])
	print('model successfully trained...')

	# test on the last 30 seconds
	test_generator = data_generator(steering_log=steering_log,
                                 image_log=image_log, 
                                 image_folder=camera_images,
                                 camera=camera,
                                 batch_size=450, # estimate of 300 x 1.5 (images per 0.1 second)
                                 image_size=image_size,
                                 timestamp_start=14751877262-300,
                                 shuffle=False,
                                 preprocess_input=input_processor,
                                 preprocess_output=output_processor)
	test_x, test_y = test_generator.next()
	print('test data shape:', test_x.shape, test_y.shape)
	yhat = model.predict(test_x)	
	rmse = np.sqrt(np.mean((yhat-test_y)**2))
	print("model evaluated RMSE:", rmse)

	plt.figure(figsize = (32, 8))
	plt.plot(test_y, 'r.-', label='target')
	plt.plot(yhat, 'b^-', label='predict')
	plt.legend(loc='best')
	plt.title("RMSE: %.2f" % rmse)
	plt.show()
	model_fullname = "%s_%d.png" % (model_name, int(time.time()))
	plt.savefig(model_fullname)


if __name__ == '__main__':
	main()