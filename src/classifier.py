from gendata import gen, create_target
import sys

def classify(perc):
    inp = gen(perc)
    train_data = inp[0]
    train_target = inp[1]
    test_data = inp[2]
    test_target = inp[3]

    #do neural network stuff

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print 'Usage: python classifier.py percentage_for_train'
    else:
        classify(float(sys.argv[1]))
