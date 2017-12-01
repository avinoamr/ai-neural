# Classification is aguably the hallmark of supervised machine learning. But
# usually our real-life examples will involve classifing into multiple classes,
# and not just a single one. One example is wanting to predict the next
# character in a text (out of the full alphabet), or the digit represented by an
# image (out of the 10 digits). In other words - we don't want a single output,
# but several mutually exclusive outputs.
#
# Even in the case where we only need to predict one class, it might be better
# to construct it with 2 classes: on and off. This is because a key problem that
# we had before is that we needed to come up with some arbitrary threshold to
# determine the binary state of the output: round(y), or y >= 0.5.  This is
# obviously error-prone and requires tuning. A much better approach will be use
# these two classes, and just check which of the two (on or off) had a higher
# activation - meaning that the model predicts a higher likelihood for, without
# any tresholding tricks. This is what we'll do next.
import numpy as np

EPOCHS = 300
STEP = 0.01
BATCHSIZE = 8

# input is the same - but the output now it's a number - it's a class that
# indicates if the passenger survived (S) or dieds (D). You can see how we can
# now add other classes (injured?) to predict.
# N = 2 (f, m) ; M = 2 (S, D)
X = ["f", "m", "f", "m", "f", "m", "f", "m", "f", "m", "f", "m", "f", "m"]
T = ["S", "D", "S", "S", "S", "D", "D", "D", "S", "D", "S", "S", "S", "D"]
data = zip(X, T)


class OneHot(object):
    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.N = len(alphabet)

    def encode(self, v):
        x = np.zeros(self.N)
        idx = self.alphabet.index(v)
        x[idx] = 1.
        return x

    # Our y-vector contains multiple values now. Out of these, we want to
    # pick one value that represents the result best in the alphabet of
    # classes. For now - we'll just pick the y-value that has the highest
    # activation. When then find the index of that value in the y-vector and
    # use that that read the class of out the alphabet.
    def decode(self, y):
        # np.argmax(y) gives us the index of the maximum value in the y-vector
        return self.alphabet[np.argmax(y)]


# create the encoder (for the input) and decoder (for target and output)
inp = OneHot(list(set(X))) # encode from gender to input neurons (x).
out = OneHot(list(set(T))) # decode from output neurons (y) to S or D.

# Each output neuron has a full set of its own weights. This is called a
# fully-connected network, because every output is connected to all inputs with
# its own separate set of weights to be learned. So, if before we needed N
# weights, now we need Mx(1 + N) weights. Each vector represents the weights of
# a single output neuron
w = np.zeros((out.N, 1 + inp.N)) # add the bias weight to encoded values
for i in xrange(EPOCHS):
    # we will again compute the accuracy - but this time without rounding. See
    # below.
    accuracy = 0
    l = 0
    for j in xrange(0, len(data), BATCHSIZE):
        minib = data[j:j+BATCHSIZE]
        dw = 0
        for v, target in minib:
            x = inp.encode(v)
            x = np.insert(x, 0, 1.) # add the fixed bias.

            # instead of computing a single result y-value for the input, we now
            # have M such values - one for every possible output class. We
            # repeat the same logic as before, only for each y-value along with
            # its weights vector.
            #
            # NOTE Same as: y = np.dot(w, x)
            y = np.zeros(out.N)
            for j in xrange(out.N):
                y[j] = sum(x * w[j])

            # did we predict correctly? Same as before, we need to transform the
            # analog output value to a binary one, per class. We no longer need
            # to use a threshold, instead we'll just pick the class assigned to
            # the neuron that exhibited the highest activation (most probable
            # predicted class). This is done by the output decoder:
            res = out.decode(y) # a string, either "S" or "D"
            accuracy += 1 if res == target else 0 # simple as that!

            # loss and derivatives
            # Same as before - only now we need to repeat the computation of
            # loss and derivatives for each y-value.
            t = out.encode(target) # encode target string to one-hot activation
            l += (y - t) ** 2 / 2 # vector of M-losses
            dy = y - t # vector of M derivatives w.r.t y, one per output value

            # before, out derivatives w.r.t the weights were a single vector of
            # N + 1 values - one per weight. But recall that now our weights are
            # an Mx(1 + N) matrix, so we need a similarily shaped matrix of
            # derivatives. Each row in the weights (and dy) corresponds to a
            # single output value, so we want to chain dy, for each y, to the
            # entire input (which is the dy/dw derivative).
            dw += np.array([dyi * x for dyi in dy]) # Mx(1 + N) derivatives

        dw /= len(minib) # average weights
        w += STEP * -dw # mini-batch update

    l = sum(l) / len(data)
    print "%s LOSS = %f ; ACCURACY = %d of %d" % (i, l, accuracy, len(data))

print
print "W = %s" % w
