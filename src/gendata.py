import numpy, os, random, sys

from math import *
from skimage import io
from skimage.transform import resize

def target(score):
	scores = numpy.arange(0.1, 1.01, 0.3)
	t = numpy.zeros(len(scores))

	for i in range(len(scores)):
		if score <= scores[i]:
			t[i] = 1
			break;

	return t

def simplify(img, pixel_density=255):
	for i in range(len(img)):
		img[i] = int(pixel_density * img[i]) / (pixel_density * 1.0)

	return img

def get_path(img_line):
	return 'BigData/' + img_line[1] + '/' + img_line[2]
def get_memorability(img_line):
	if(float(img_line[4]) == 0.0):
		return float(img_line[4])
	else:
		return float(img_line[3]) / (float(img_line[3]) + float(img_line[4]))

def gen(perc, max_images):

	data = []
	helper = []
	train_data = []
	train_target = []
	test_data = []
	test_target = []

	with open('BigData/memorabiliy.csv') as f:
		content = f.readlines()

	content = content[1:min(len(content), max_images)]

	print 'Loading images..'
	for info in content:
		aux = info.rstrip('\n').split(',')
		helper.append([int(aux[0])-1, get_memorability(aux), aux[1]+'/'+aux[2]])

		img = io.imread(get_path(aux), as_grey=True)
		img = resize(img, (64,64))
		img = img.ravel()
		img = simplify(img, pixel_density=8)

		data.append(img)

	print 'Image loading finished. Generating train and test data..'
	random.shuffle(helper)
	ind = int(floor(len(helper)*perc))

	for i in range(ind):
		train_data.append(data[helper[i][0]])
		train_target.append(target(helper[i][1]))
	for i in range(ind, len(helper)):
		test_data.append(data[helper[i][0]])
		test_target.append(target(helper[i][1]))

	return [train_data, train_target, test_data, test_target, helper]