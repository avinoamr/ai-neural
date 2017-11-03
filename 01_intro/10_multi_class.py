# Classification is aguably the hallmark of supervised machine learning. But
# usually our real-life examples will involve classifing into multiple classes,
# and not just a single one. One example is wanting to predict the next
# character in a text (out of the full alphabet), or the digit represented by an
# image (out of the 10 digits). In other words - we don't want a single output,
# but several mutually exclusive outputs.
import csv
import random
import numpy as np

STEP = .001
EPOCHS = 300

# Say we want to build a simple program to dechiper code. We're given a few
# words in the input and their corresponding dechipered text as output.
X = "croj dsmujlayfxjpygjtdwzbjyeoajcrojvkihjnyq"
T = "the quick brown fox jumps over the lazy dog"

# we'll use one-of-k encoding for each input character. One neuron per character
# in the alphabet of the provided text
data = zip(X, T) # zip the input and output together

ALPHABET = list(set(X + T)) # list of all unique characters
BATCHSIZE = len(data) / 4
N = 1 + len(ALPHABET) # one neuron per input character, plus bias

# Similarily, our output will also be one-of-k. If we want to map into M
# categories/classes, we'll need M output neurons, where each is computed
# exactly like we did before.
M = len(ALPHABET) # One neuron per output character in the alphabet

# Each output neuron has a full set of its own weights. This is called a
# fully-connected network, because every output is connected to all inputs with
# its own separate set of weights to be learned. So, if before we needed N
# weights, now we need M * N weights. We'll achieve it by creating an MxN matrix
# of weights where each vector represents the weights of a single output neuron
w = np.zeros((M, N)) # - .5

# given a single character, return all of the alphabet neurons with the right
# one lit up
def one_of_k(c):
    x = np.zeros(len(ALPHABET))
    idx = ALPHABET.index(c)
    x[idx] = 1. # current character
    return x

for i in xrange(EPOCHS):
    l = 0
    for c0, c1 in data:
        x = one_of_k(c0)
        x = np.insert(x, 0, 1.) # bias
        t = one_of_k(c1)

        y = np.zeros(M)
        for j in xrange(M):
            y[j] = sum(x * w[j])
            l = (y[j] - t[j]) ** 2 / 2
            dw = (y[j] - t[j]) * x
            w[j] += STEP * -dw

    if i % 100 == 0:
        print i

# print zip(ALPHABET, w[0][1:])
print
for j in xrange(M):
    c0 = ALPHABET[j]
    x = one_of_k(c0)
    x = np.insert(x, 0, 1.) # bias

    # print "INPUT: %s" % c0

    y = np.zeros(M)
    for i in xrange(M):
        y[i] = sum(x * w[i])
        # print ALPHABET[i], sum(x * w[i])

    # print
    # print y, np.argmax(y), ALPHABET[np.argmax(y)]
    c1 = ALPHABET[np.argmax(y)]

    idx = X.index(c0)

    print "%s = %s %d" % (c0, c1, T[idx] == c1)


X = "scjfyaub"
result = ""
for c0 in X:
    x = one_of_k(c0)
    x = np.insert(x, 0, 1.) # bias

    y = np.zeros(M)
    for i in xrange(M):
        y[i] = sum(x * w[i])

    result += ALPHABET[np.argmax(y)]

print result
