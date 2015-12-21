from __future__ import print_function
from gendata import *

import numpy, sys

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

def ConvolutionalNeuralNetwork(train_data, train_target, test_data, test_target):
	print('cnn')
	
def MultiLayerPerceptron(train_data, train_target, test_data, test_target):
	numpy.random.seed(1337)

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

	rms = RMSprop()
	model.compile(loss='categorical_crossentropy', optimizer=rms)
	model.fit(train_data, train_target,
			batch_size=batch, nb_epoch=epoch,
			show_accuracy=True, verbose=2,
			validation_data=(test_data, test_target))
	score = model.evaluate(test_data, test_target,
			show_accuracy=True, verbose=1)

	print('Test score:', score[0])
	print('Test accuracy:', score[1])

	return model.predict(test_data, verbose=0)

def perft(out, target, helper):
	print(os.linesep + 'Detailed prediction:')

	for i in range(len(out)):
		print('Expected: ' + str(numpy.argmax(target[i])) + ' ' +
			'- Predicted: '+ str(numpy.argmax(out[i])) + ' ' +
			'- Image: '    + str(helper[i][2]))

def classify(perc, max_images):
	inp = gen(perc, max_images)

	train_data   = numpy.array(inp[0])
	train_target = numpy.array(inp[1])
	test_data    = numpy.array(inp[2])
	test_target  = numpy.array(inp[3])
	helper 		 = numpy.array(inp[4])

	out = MultiLayerPerceptron(train_data, train_target, test_data, test_target)
	perft(out, test_target, helper)

if __name__ == "__main__":
	if len(sys.argv) != 3:
		print('Usage: python new_classifier.py percentage_for_train max_images')
	else:
		classify(float(sys.argv[1]), int(sys.argv[2]))