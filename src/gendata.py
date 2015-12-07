import numpy, os
from skimage import io
from skimage.transform import resize

#shape retorna height, width

def gen(perc):

	data = []
	scores = []
	max_w = 0
	max_h = 0
	train_data = []
	train_target = []
	test_data = []
	test_target = []

	with open('scores.txt') as f:
		content = f.readlines()

	for info in content:
		aux = info[0:len(info)-1].split(',')
		scores.append(float(aux[1]))
		img = io.imread(os.getcwd() + '/Data/' + aux[0], True)
		data.append(img)
		max_h = max(max_h, img.shape[0])
		max_w = max(max_w, img.shape[1])

	ind = 0
	for img in data:
		print("Processing ...")
		test_data.append(resize(img, (max_h, max_w)))
		#test_data[ind].append(scores[ind])
		ind += 1

if __name__ == "__main__":
	gen(0.7)
