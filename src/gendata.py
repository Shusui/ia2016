import numpy, os
from skimage import io
from sklearn.decomposition import sparse_encode, MiniBatchDictionaryLearning

def gen(batch_iter=100):

	feats = []
	max_components = 0
	
	imagesfolder = os.getcwd() + "/Data/"
	imagespaths = os.listdir(imagesfolder)

	for path in imagespaths:
		img = io.imread(imagesfolder + path, True)
		feats.append(img)

	mbdl = MiniBatchDictionaryLearning(n_components=len(imagespaths), n_iter=batch_iter)
	
	for feat in feats:
		mbdl.fit(feat)

	"""
	for feat in feats:
		code = sparse_encode(feat, mbdl.components_, alpha=0.1)
		print code, "\n", len(code), len(code[0])
	"""

if __name__ == "__main__":
	numpy.set_printoptions(precision=3, suppress=True)
	gen(10)