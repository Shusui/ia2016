from gendata import gen, create_target
from DBN import *
import sys, numpy

def fit(train_data, train_target, pretrain_epochs=100, pretrain_lr=0.1, finetune_epochs=20, finetune_lr=0.1):
	n_ins  = len(train_data[0])
	n_outs = len(train_target[0])

	dbn = DBN(input=train_data, label=train_target, n_ins=n_ins, n_outs=n_outs)
	dbn.pretrain(lr=pretrain_lr, epochs=pretrain_epochs, verbose=True)
	dbn.finetune(lr=finetune_lr, epochs=finetune_epochs, verbose=True)

	return dbn

def perft(dbn, test_data, test_target, to_shuffle, verbose=False):
	out = dbn.predict(test_data)
	if verbose:
		print 'Deep belief net output:\n', out

	size = len(out)
	perf = 0.0
	for i in range(size):
		if numpy.argmax(out[i]) == numpy.argmax(test_target[i]):
			perf += 1

		if verbose:
			img_name = str(int(to_shuffle[len(to_shuffle)-size+i][0])) + '.jpg'
			print 'Image:', img_name, '\t','Expected:', numpy.argmax(test_target[i]), '- Predicted:', numpy.argmax(out[i])

	print 'Performance rate: ', perf / len(out) * 100

def classify(perc):
	inp = gen(perc)

	train_data   = numpy.array(inp[0])
	train_target = numpy.array(inp[1])
	test_data    = numpy.array(inp[2])
	test_target  = numpy.array(inp[3])
	to_shuffle	 = numpy.array(inp[4])

	dbn = fit(train_data, train_target, 1000, 0.1, 200)
	perft(dbn, test_data, test_target, to_shuffle, True)

if __name__ == "__main__":
	if len(sys.argv) != 2:
		print 'Usage: python classifier.py percentage_for_train'
	else:
		classify(float(sys.argv[1]))
