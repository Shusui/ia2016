import numpy, os, random, sys

from math import *
from skimage import io
from skimage.transform import resize

def target(score):
	scores = numpy.arange(0.5, 1.01, 0.125)
	t = numpy.zeros(len(scores))

	for i in range(len(scores)):
		if score <= scores[i]:
			t[i] = 1
			return t

def mem_pos(score):
	scores = numpy.arange(0.5, 1.01, 0.125)

	for i in range(len(scores)):
		if score <= scores[i]:
			return i

def simplify(img, pixel_density=255):
	for i in range(len(img)):
		for j in range(len(img[0])):
			if pixel_density <= 2:
				img[i][j] = 0.0 if img[i][j] < 0.5 else 1.0
			else:
				img[i][j] = int(pixel_density * img[i][j]) / (pixel_density * 1.0)

	return img

def get_path(aux):
	return 'lamem/' + aux[1]

def get_mem(aux):
	return float(aux[2])

def print_status(number, path, req):
	sys.stdout.write("\r\x1b[KLoading image "+path[3:]+" - Status: "+str(int(number)+1)+"/"+str(req))
	sys.stdout.flush()

def gen(perc, max_images, dimension=2):

	data = []
	helper = []
	train_data = []
	train_target = []
	test_data = []
	test_target = []

	displacement = [0,0,0,0,0]

	with open('memorability.txt') as f:
		content = f.readlines()

	request = min(len(content), max_images)
	content = content[1:request]

	print 'Loading images.. please wait.'
	for info in content:
		aux = info.rstrip('\n').split(',')
		mem = get_mem(aux)

		print_status(aux[0], aux[1], request)

		displacement[mem_pos(mem)] += 1
		helper.append([int(aux[0])-1, mem, get_path(aux)])

		img = io.imread(get_path(aux), as_grey=True)
		img = resize(img, (256,256))

		# reduce pixel density to simplify processing
		# img = simplify(img, pixel_density=2)

		if dimension == 2:
			img = img.ravel()

		data.append(img)

	print os.linesep, 'Image loading finished. Generating train and test data..'
	random.shuffle(helper)
	ind = int(floor(len(helper)*perc))

	for i in range(ind + (len(helper)-ind)/2):
		train_data.append(data[helper[i][0]])
		train_target.append(target(helper[i][1]))
	for i in range(ind, len(helper)):
		test_data.append(data[helper[i][0]])
		test_target.append(target(helper[i][1]))

	print 'Data displacement:',
	for i in range(len(displacement)):
		print displacement[i],

	print os.linesep, 'Data generated successfully.'
	return [numpy.array(train_data), numpy.array(train_target),
			numpy.array(test_data),  numpy.array(test_target), helper, ind]