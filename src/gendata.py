import numpy, os, random
from math import floor
from skimage import io
from skimage.transform import resize

def target(score):
	t = numpy.zeros(10)
	t[int(floor(score * 10))] = 1
	return t

def gen(perc):

	data = []
	to_shuffle = []
	train_data = []
	train_target = []
	test_data = []
	test_target = []

	max_images = 4

	print "Loading images.."
	with open('scores.txt') as f:
		content = f.readlines()

	content = content[:max_images]

	for info in content:
		aux = info.rstrip('\n').split(',')
		to_shuffle.append([int(aux[0][:-4])-1, float(aux[1])])

		img = io.imread('Data/' + aux[0], as_grey=True)
		img = resize(img, (16,16))
		img = img.ravel()

		data.append(img)
	print "All images loaded and normalized.."

	random.shuffle(to_shuffle)
	ind = int(floor(len(to_shuffle)*perc))

	print "Getting train and test data.."

	for i in range(ind):
		train_data.append(data[to_shuffle[i][0]])
		train_target.append(target(to_shuffle[i][1]))
	for i in range(ind, len(to_shuffle)):
		test_data.append(data[to_shuffle[i][0]])
		test_target.append(target(to_shuffle[i][1]))
	print "Ready for classification.."

	return [train_data, train_target, test_data, test_target, to_shuffle]
