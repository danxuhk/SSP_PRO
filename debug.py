import os, sys
sys.path.append(os.getcwd())

import time
import tflib as lib
import tflib.save_images
import tflib.mnist
import tflib.cifar10
import tflib.plot
import tflib.inception_score
from scipy.stats import rv_discrete
import numpy as np

DATA_DIR = '/data2/danxu/image_generation/cifar-10-batches-py/'
MODE = 'wgan-gp' # Valid options are dcgan, wgan, or wgan-gp
DIM = 128 # This overfits substantially; you're probably better off with 64
LAMBDA = 10 # Gradient penalty lambda hyperparameter
CRITIC_ITERS = 5 # How many critic iterations per generator iteration
BATCH_SIZE = 64 # Batch size
ITERS = 200000 # How many generator iterations to train for
OUTPUT_DIM = 3072 # Number of pixels in CIFAR10 (3*32*32)
def softmax(x):
     """Compute softmax values for each sets of scores in x."""
     return np.exp(x) / np.sum(np.exp(x), axis=0)

ratio = 0.5
elements = np.arange(30);
elements += 1;
probability_array = elements;
index = np.arange(len(probability_array));
probability_array_01 = softmax(probability_array);
len_index = len(probability_array_01);
#while (len(index_) / float(len_index)) <= ratio:
#print len_index
#print int(ratio*len_index)
#print probability_array_01
sample = np.random.choice(len_index, int(ratio*len_index), replace=False, p=probability_array_01);
print sample
    #sample = rv_discrete(values=(index, probability_array_01)).rvs(size=1)
    #index_.append(s)aample[0]);
    #indexoices[aample[0]] = -1;
    #probability_array_01[sample[0]] = 0;
    #probability_array_01 = softmax(probability_array_01);
    #if len(index_) == 10:
	#print index_
