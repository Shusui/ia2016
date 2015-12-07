import numpy, os, random
from math import floor
from skimage import io
from skimage.transform import resize

def create_target(score):
	t = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	t[int(floor(score * 10))] = 1
	return t

def gen(perc):

	data = []
	to_shuffle = []
	max_w = 0
	max_h = 0
	new_data = []
	train_data = []
	train_target = []
	test_data = []
	test_target = []

	with open('scores.txt') as f:
		content = f.readlines()

	for info in content:
		aux = info[0:len(info)-1].split(',')
		to_shuffle.append([int(aux[0].split('.')[0]) - 1, float(aux[1])])
		img = io.imread(os.getcwd() + '/Data/' + aux[0], True)
		max_h = max(max_h, img.shape[0])
		max_w = max(max_w, img.shape[1])
	print "Finished loading images..."

	for i in to_shuffle:
		img = io.imread(os.getcwd() + '/Data/' + str(i[0] + 1) + '.jpg', True)
		img = resize(img, (max_h, max_w))
		img = img.ravel()
		new_data.append(img)
	print "All images normalized..."

	data = new_data
	random.shuffle(to_shuffle)
	ind = int(floor(len(to_shuffle)*perc))
	print "Geting train and test data..."

	for i in range(0, ind):
		train_data.append(data[to_shuffle[i][0]])
		train_target.append(create_target(to_shuffle[i][1]))
	for i in range(ind, len(to_shuffle)):
		test_data.append(data[to_shuffle[i][0]])
		test_target.append(create_target(to_shuffle[i][1]))
	print "Ready for classification..."

	output = [train_data, train_target, test_data, test_target]
	return output

if __name__ == "__main__":
	gen(0.7)
