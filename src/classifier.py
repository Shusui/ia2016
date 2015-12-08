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

def perft(dbn, test_data, test_target):
	out = dbn.predict(test_data)

	perf = 0.0
	for i in range(len(out)):
		if numpy.argmax(out[i]) == numpy.argmax(test_target[i]):
			perf += 1

	print 'Performance rate: ', perf / len(out)

def classify(perc):
	inp = gen(perc)

	train_data   = numpy.array(inp[0])
	train_target = numpy.array(inp[1])
	test_data    = numpy.array(inp[2])
	test_target  = numpy.array(inp[3])

	dbn = fit(train_data, train_target)
	perft(dbn, test_data, test_target)

if __name__ == "__main__":
	if len(sys.argv) != 2:
		print 'Usage: python classifier.py percentage_for_train'
	else:
		classify(float(sys.argv[1]))
