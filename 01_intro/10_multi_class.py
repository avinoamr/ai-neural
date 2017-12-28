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
np.random.seed(1)

EPOCHS = 300
ALPHA = 0.01

# input is the same - but the output now isn't a number - it's a class that
# indicates if the passenger survived (S) or dieds (D). You can see how we can
# now add other classes (injured?) to predict.
# N = 2 (f, m) ; M = 2 (S, D)
X = [["f"], ["m"], ["f"], ["m"], ["f"], ["m"], ["f"], ["m"], ["f"], ["m"]]
T = [["S"], ["D"], ["S"], ["S"], ["S"], ["D"], ["D"], ["D"], ["S"], ["D"]]
INPUTS = ["m", "f"]
OUTPUTS = ["S", "D"]

# OneHot encodes data of arbitrary features into a list of one-hot neuron
# activations, each either a zero or one.
class OneHot(list):
    def encode(self, data):
        x = np.zeros((len(data), len(self)))
        for i, vs in enumerate(data):
            indices = [self.index(v) for v in sorted(vs)]
            x[i][indices] = 1.
        return x

# create the encoder (for the input) and decoder (for target and output)
X = OneHot(INPUTS).encode(X) # encode from gender to input neurons (x).
T = OneHot(OUTPUTS).encode(T) # decode from output neurons (y) to S or D.
data = zip(X, T)

# Each output neuron has a full set of its own weights. This is called a
# fully-connected network, because every output is connected to all inputs with
# its own separate set of weights to be learned. So, if before we needed N + 1
# weights, now we need Mx(1 + N) weights. Each vector represents the weights of
# a single output neuron
w = np.random.random((len(OUTPUTS), 1 + len(INPUTS))) # add the bias weight to encoded values
for i in xrange(EPOCHS):
    np.random.shuffle(data)

    # we will again compute the accuracy - but this time without rounding.
    accuracy = 0
    l = 0
    for x, t in data:
        x = np.insert(x, 0, 1.) # add the fixed bias.

        # instead of computing a single result y-value for the input, we now
        # have M such values - one for every possible output class. We
        # repeat the same logic as before, only for each y-value along with
        # its weights vector.
        #
        # NOTE Same as: y = np.dot(w, x)
        y = np.zeros(len(OUTPUTS))
        for j in xrange(len(OUTPUTS)):
            y[j] = sum(x * w[j])

        # loss and derivatives
        # Same as before - only now we need to repeat the computation of
        # loss and derivatives for each y-value.
        l += (y - t) ** 2 / 2 # vector of M-losses
        dy = y - t # vector of M derivatives w.r.t y, one per output value

        # before, out derivatives w.r.t the weights were a single vector of
        # N + 1 values - one per weight. But recall that now our weights are
        # an Mx(1 + N) matrix, so we need a similarily shaped matrix of
        # derivatives. Each row in the weights (and dy) corresponds to a
        # single output value, so we want to chain dy, for each y, to the
        # entire input (which is the dy/dw derivative).
        #
        # NOTE Same as: np.array([d * x for d in dy])
        dw = np.zeros(w.shape) # Mx(1 + N) derivatives
        for j, d in enumerate(dy):
            dw[j] = d * x

        # updare
        w += ALPHA * -dw # mini-batch update

        # did we predict correctly? Same as before, we need to transform the
        # analog output value to a binary one, per class. We no longer need
        # to use a threshold, instead we'll just pick the class assigned to
        # the neuron that exhibited the highest activation (most probable
        # predicted class). This is done by the output decoder:
        accuracy += 1 if np.argmax(y) == np.argmax(t) else 0 # simple as that!

    l = sum(l) / len(data)
    print "%s LOSS = %f ; ACCURACY = %d of %d" % (i, l, accuracy, len(data))

print
print "W = %s" % w
