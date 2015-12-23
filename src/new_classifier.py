from __future__ import print_function
from gendata import *

import numpy, sys

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

def ConvolutionalNeuralNetwork(perc, max_images):
	train_data, train_target, test_data, test_target, helper = gen(perc, max_images, 3)

	epoch  = 10
	batch  = 128
	layers = 128

	n_out  = train_target.shape[1]
	img_rows = train_data.shape[1]
	img_cols = train_data.shape[2]

	train_data = train_data.reshape(train_data.shape[0], 1, img_rows, img_cols)
	test_data  = test_data.reshape(test_data.shape[0], 1, img_rows, img_cols)

	filt = 8
	conv = 3
	pool = 2

	model = Sequential()

	model.add(Convolution2D(filt, conv, conv, border_mode='valid', input_shape=(1, img_rows,img_cols)))
	model.add(Activation('relu'))
	model.add(Convolution2D(filt, conv, conv))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(pool, pool)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(layers))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(n_out))
	model.add(Activation('softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='adadelta')
	model.fit(train_data, train_target,
		batch_size=batch, nb_epoch=epoch,
		show_accuracy=True, verbose=1,
		validation_data=(test_data, test_target))

	score = model.evaluate(test_data, test_target,
		show_accuracy=True, verbose=1)

	print('Test score:', score[0])
	print('Test accuracy:', score[1])

	out = model.predict(test_data, verbose=0)
	perft(out, test_target, helper)

def MultiLayerPerceptron(perc, max_images):
	train_data, train_target, test_data, test_target, helper = gen(perc, max_images, 2)

	epoch  = 100
	batch  = 128
	layers = 256

	n_ins  = len(train_data[0])
	n_out  = len(train_target[0])

	model = Sequential()
	model.add(Dense(layers, input_shape=(n_ins,)))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	model.add(Dense(layers))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	model.add(Dense(n_out))
	model.add(Activation('softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='adadelta')
	model.fit(train_data, train_target,
			batch_size=batch, nb_epoch=epoch,
			show_accuracy=True, verbose=2,
			validation_data=(test_data, test_target))

	score = model.evaluate(test_data, test_target,
			show_accuracy=True, verbose=1)

	print('Test score:', score[0])
	print('Test accuracy:', score[1])

	out = model.predict(test_data, verbose=0)
	perft(out, test_target, helper)

def perft(out, target, helper):
	print(os.linesep + 'Detailed prediction:')

	for i in range(len(out)):
		print('Expected: ' + str(numpy.argmax(target[i])) + ' ' +
			'- Predicted: '+ str(numpy.argmax(out[i])) + ' ' +
			'- Image: '    + str(helper[i][2]))

def classify(perc, max_images):
	numpy.random.seed(1337)

	#MultiLayerPerceptron(perc, max_images)
	ConvolutionalNeuralNetwork(perc, max_images)

if __name__ == "__main__":
	if len(sys.argv) != 3:
		print('Usage: python new_classifier.py percentage_for_train max_images')
	else:
		classify(float(sys.argv[1]), int(sys.argv[2]))