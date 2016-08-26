
# coding: utf-8

# ## Dowload and extract weight parameters of AlexNet
# http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy
# Simple script to extract weights of trained AlexNet
# @author: Swaroop Guntupalli <swaroopgj@gmail.com>

# In[2]:

import urllib
import numpy as np
import tensorflow as tf
import os, sys


# In[3]:

def get_data():
    fname = 'bvlc_alexnet.npy'
    if not os.path.exists(fname):
        url = 'http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy'
        urllib.urlretrieve(url, fname)
    return np.load(fname).item()

net_params = get_data()

print net_params.keys()


def get_weights(key):
    """
    Returns a tuple (weights, biases)
    key can be one of [u'fc6', u'fc7', u'fc8', u'conv3', u'conv2', u'conv1', u'conv5', u'conv4']
    """
    return (net_params.get(key)[0], net_params.get(key)[1])

print "Example weights shape for fc8", get_weights('fc8')[0].shape
print "Example weights shape for conv1", get_weights('conv1')[0].shape

if __name__ == '__main__':
    if len(sys.argv) == 2:
        key = sys.argv[1]
    else:
        key = 'fc8'
    print '%s weights'%(key), get_weights(key)[0]
