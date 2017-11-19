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
EPOCHS = 100

# Say we want to build a simple program to dechiper code. We're given a few
# words in the input and their corresponding dechipered text as output.
X = "croj dsmujlayfxjpygjtdwzbjyeoajcrojvkihjnyq*"
T = "the quick brown fox jumps over the lazy dog!"

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
def encode(c):
    x = np.zeros(len(ALPHABET))
    idx = ALPHABET.index(c)
    x[idx] = 1. # current character
    return x

# Our y-vector contains M values now. Out of these, we want to pick one value
# that represents the result best in the alphabet of classes. For now - we'll
# just pick the y-value that has the highest activation. When then find the
# index of that value in the y-vector and use that that read the class of out
# the alphabet.
def decode(y):
    # np.argmax(y) gives us the index of the maximum value in the y-vector
    return ALPHABET[np.argmax(y)]

# perform a prediction, returning the predicted, decoded value. If a target is
# supplied, it also returns the loss and derivatives. We have now separated the
# prediction process into its own function for re-usability
def predict(v, target = None):
    x = encode(c0)
    x = np.insert(x, 0, 1.) # bias

    # instead of computing a single result y-value for the input, we now have
    # M such values - one for every possible output class. We repeat the same
    # logic as before, only for each y-value along with its weights vector.
    # NOTE Equivalent one-liner: y = np.dot(w, x)
    y = np.zeros(M)
    for j in xrange(M):
        y[j] = sum(x * w[j])

    res = decode(y) # and decode back into the class
    if target is None:
        return res

    # loss and derivatives
    t = encode(target)

    # Same as before - only now we need to repeat the computation of loss and
    # derivatives for each y-value.
    # NOTE Equivalent one-lines:
    #       l = (y - t) ** 2 / 2 ; dw = ...?
    #       dw = np.array([d * x for d in y - t])
    l = np.zeros(M) # M losses.
    dw = np.zeros((M, N)) # MxN derivatives - one for every weight
    for j in xrange(len(y)):
        l[j] = (y[j] - t[j]) ** 2 / 2
        dw[j] = (y[j] - t[j]) * x

    l = sum(l) / len(l) # average the loss over all of the outputs/targets
    return res, l, dw

# learn the weights
for i in xrange(EPOCHS):
    l = 0
    accuracy = 0
    for c0, c1 in data:
        c2, l0, dw = predict(c0, c1)
        l += l0
        w += STEP * -dw
        accuracy += 1 if c2 == c1 else 0

    print "%s: LOSS = %s; ACCURACY = %d of %d" % (i, l, accuracy, len(data))

# decipher another message
X = "scjfyaub*"
result = ""
for c0 in X:
    result += predict(c0)

print
print X + " = " + result
print
