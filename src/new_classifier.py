from __future__ import print_function
from gendata import *

import numpy, sys, math

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

def ConvolutionalNeuralNetwork(perc, max_images):
	train_data, train_target, test_data, test_target, helper, ind = gen(perc, max_images, 3)

	epoch  = 10
	batch  = 128
	# hidden units
	layers = 256

	n_out  = train_target.shape[1]
	img_rows = train_data.shape[1]
	img_cols = train_data.shape[2]

	train_data = train_data.reshape(train_data.shape[0], 1, img_rows, img_cols)
	test_data  = test_data.reshape(test_data.shape[0], 1, img_rows, img_cols)

	filt = 16
	conv = 3
	pool = 2

	model = Sequential()

	model.add(Convolution2D(filt, conv, conv, border_mode='valid', input_shape=(1, img_rows,img_cols)))
	model.add(Activation('relu'))
	model.add(Convolution2D(filt, conv, conv))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(pool, pool)))
	model.add(Dropout(0.25))

	'''
	model.add(Convolution2D(2*filt, conv, conv, border_mode='valid'))
	model.add(Activation('relu'))
	model.add(Convolution2D(2*filt, conv, conv))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(pool, pool)))
	model.add(Dropout(0.25))
	'''

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

	print('Validation score:', score[0])
	print('Validation accuracy:', score[1])

	out = model.predict(test_data, verbose=0)
	perft(out, test_target, helper, score[1], ind)

def MultiLayerPerceptron(perc, max_images):
	train_data, train_target, test_data, test_target, helper, ind = gen(perc, max_images, 2)

	epoch  = 100
	batch  = 128
	# hidden units
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

	print('Validation score:', score[0])
	print('Validation accuracy:', score[1])

	out = model.predict(test_data, verbose=0)
	perft(out, test_target, helper, score[1], ind)

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def perft(out, target, helper, test_acc, ind, real_acc=0.0):
	print(os.linesep + 'Detailed prediction:')

	for i in range(len(out)):
		expected  = numpy.argmax(target[i])
		predicted = numpy.argmax(out[i])

		print('Expected: ' + str(expected) + ' ' +
			'- Predicted: '+ str(predicted) + ' ' +
			'- Image: '    + str(helper[ind+i][2]))

		if expected == predicted:
			real_acc += 1

	assert(isclose(float(acc) / len(out), test_acc))

def classify(perc, max_images):
	numpy.random.seed(1337)

	#MultiLayerPerceptron(perc, max_images)
	ConvolutionalNeuralNetwork(perc, max_images)

if __name__ == "__main__":
	if len(sys.argv) != 3:
		print('Usage: python new_classifier.py percentage_for_train max_images')
	else:
		classify(float(sys.argv[1]), int(sys.argv[2]))