# One conceptual problem some people face when using multiple layers, is
# understanding the intuition around what kind of capabilities it adds to the
# network - especially under certain configurations - like the dimensions and
# number of hidden layers. We will explore that here.
#
# Essentially - having 2-layers network (input to hidden, hidden to output), is
# exactly the same as having 1-layer (input to output) except that we have some
# arbitrary code modify our input. So it's the same kind of logic, but with a
# different input. This means that we can think of adding hidden layers just as
# a process of transforming the input in some way. The broader term here is
# feature-selection.
#
# The features of our data are the encoded inputs that are fed into the network.
# In the titanic example, we had some hand-selected features. You can imagine
# that that the original records from the titanic had a lot more information
# than the limited set we've used. So someone had to manually pick which
# features likely matter the most and input this sub-set of the data into our
# training and test set. This is obviously a huge limitation because it requires
# some assumptions about the correlation between the input and output. Imagine
# problems that involve image or video recognition where the input may contain
# 1024x1024 pixels as input. It's far more difficult to choose a subset in these
# cases. Of course - we could just use all possible inputs, resulting in much
# slower and more complicated learning - but it will work. For linear cases.
#
# Another, perhaps more complicated challenge of feature selection - is that
# features may be more abstract - like some combinations of other features in
# the data. This means that the correlation with the output may be non-linear.
# For example - in the titanic example, we might want to distinguish Old-Female
# as a separate feature because it's not just a linear summation of the two
# separate features: Old + Female. It's possible that the rules (weights) of
# Old-Female passengers are completely different (perhaps even the opposite of)
# Old + Female. One such speculation is that maybe if the passenger was female
# and old she's perhaps even more likely to survive than if she's female + kid.
# This means that in order to support that with one layer, we'll need every
# possible combinations of all features - exploding the dataset.
#
# NOTE that in the previous exercise we took a detour away from the fully-fleged
# learning procedure in favor of focusing on the back-prop algorithm for one
# specific instance. In this example we'll combine the back-prop algorithm with
# our full learning process.
import numpy as np
import csv
import sys
np.random.seed(1)

ALPHA = .1
EPOCHS = 100
EPSILON = 0.0001

# In this dataset we will attempt to learn the same data using different
# configurations and then inspect the weights produced by the hidden layer to
# understand what was encoded by that layer.
X = []
T = []

# Layer represents a single neural network layer of weights
class Layer(object):
    W = None
    _last = (None, None) # input, output

    def __init__(self, w):
        self.W = w

    # forward pass is the same as before.
    def forward(self, x):
        x = np.append(x, 1.) # add the fixed input for bias
        y = np.dot(self.W, x) # derivate: x
        self._last = x, y
        return y

    # backward pass - compute the derivatives of each weight in this layer and
    # update them, returning the derivatives of the input to this layer
    def backward(self, dy):
        x, y = self._last
        dw = np.array([d * x for d in dy])
        dx = np.dot(dy, self.W)
        # self.W -= ALPHA * dw # update
        return np.delete(dx, -1), dw # remove the bias derivative

# read the data from the CSV file
data = [d for d in csv.DictReader(open("09_titanic.csv"))]
N = 20
BATCHSIZE = len(data) / 4

vocabs = {
    "Fare": { "cheap": 0, "low": 1, "medium": 2, "high": 3 },
    "Embarked": { "S": 4, "C": 5, "Q": 6 },
    "Age": { "kid": 7, "young": 8, "adult": 9, "old": 10 },
    "Family": { "alone": 11, "small": 12, "medium": 13, "big": 14 },
    "Pclass": { "1": 15, "2": 16, "3": 17 },
    "Sex": { "male": 18, "female": 19 }
}

# encode the data into N input neurons
def encode(d):
    x = np.zeros(N)
    for k, v in vocabs.items():
        idx = v[d[k]]
        x[idx] = 1.

    return x

l1 = Layer(np.random.random((1, N + 1)))
l2 = Layer(np.random.random((2, 1 + 1)))

# predict the output and loss for the given input and target
def predict(x, t):
    h = l1.forward(x)
    y = l2.forward(h) # output from first layer is fed as input to the second

    # now compute our error, same as before
    e = (y - t) ** 2 /2
    return y, e

# learn the weights in all layers
for i in xrange(EPOCHS):
    np.random.shuffle(data)
    l = 0
    accuracy = 0

    for j in xrange(0, len(data), BATCHSIZE):
        minib = data[j:j+BATCHSIZE]
        dw1 = 0
        dw2 = 0

        # print "l1.W=", l1.W
        # print "l2.W=", l2.W
        for d in minib:
            x = encode(d) # encode the input features into multiple 1-of-key's
            t = np.zeros(2)
            t[int(d["Survived"])] = 1.
            y, e = predict(x, t)
            l += e

            dWs = [0, 0] # derivatives of all weights in both layers.
            # Ws = [l1.W, l2.W] # all of the weights in the network
            # for j, w in enumerate(Ws): # iterate over all weight matrices in the network
            #     dW = dWs[j] = np.zeros(w.shape)
            #
            #     # for every weight - re-run the entire network after applying a tiny change
            #     # to that weight in order to measure how it affects the total loss.
            #     for i in range(len(w)):
            #         for j in range(len(w[i])):
            #             w[i][j] += EPSILON # add a tiny epsilon amount to the weight
            #             _, e_ = predict(x, t) # re-run our network to predict the new error
            #             dW[i][j] = sum(e_ - e) / EPSILON
            #             w[i][j] -= EPSILON # revert our change.

            d, dWs[1] = l2.backward(y - t)
            _, dWs[0] = l1.backward(d)

            dw1 += dWs[0]
            dw2 += dWs[1]

            idx = np.argmax(y)
            if t[idx] == 1.:
                accuracy += 1
            # accuracy += 1 if round(np.clip(y, 0, 1)) == t else 0

        # print dw1
        dw1 /= len(minib)
        dw2 /= len(minib)

        # print dw1

        # print "dw1=", dw1
        # print "dw2=", dw2
        l1.W += ALPHA * -dw1 # mini-batch update
        l2.W += ALPHA * -dw2 # mini-batch update

    print "%s: LOSS = %s; ACCURACY = %d of %d" % (i, l, accuracy, len(data))
    # print "l1.W=", l1.W
    # print "l2.W=", l2.W
