# We will now try to apply everything we've learned to a real-life example:
# predicting the liklihood of passengers aboard the RMS Titanic. This is the
# introductory competition in Kaggle[1] and provides a simple first step in the
# world of machine learning.
#
# The data provided (titanic.csv) contains almost 900 passenger information -
# including Sex, Age, ticket class, cabin, etc., as well as an indication
# if the passenger survived or not. NOTE that the actual data from the
# competition is a bit messy, as it contains missing fields and a  mixture of
# scalar, ordinal and categorical values. There's a great deal of preprocessing
# involved in getting this data ready for analysis, but we'll skip it for now.
# You can learn more by reading through some of the kernels in the competition.
# Instead, our data here is pre-cleaned and ready for use.
#
# Our goal is to be able to predict, given all of the parameters mentioned,
# if the passenger survived or not.
#
# [1] https://www.kaggle.com/c/titanic
import csv
import numpy as np
import copy
np.random.seed(1)

STEP = 1.8
EPOCHS = 25000000
H = 30

# read the data from the CSV file and break the data into an input and output
# sets, each is a list of (k,v) tuples
data = [d for d in csv.DictReader(open("30_titanic.csv"))]
T = [("Survived", d.pop("Survived")) for d in data]
X = [d.items() for d in data]
data = zip(X, T)

inp_vocab = set()
for d in X:
    inp_vocab.update(d)

out_vocab = set(T)

# we have a lot of noise - if you try a batchsize of 1, you'll see that it takes
# a huge amount of time to converge. Other methods, like adapatable leanring
# rate can also work around that issue, arguably in a more generic and robust
# way.
BATCHSIZE = len(data) #  / 4

class OneHot(object):
    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.N = len(alphabet)

    def encode(self, *vs):
        x = np.zeros(self.N)
        for v in vs:
            try:
                idx = self.alphabet.index(v)
                x[idx] = 1.
            except ValueError:
                pass
        return x

    def decode(self, y):
        return self.alphabet[np.argmax(y)]

# Layer represents a single neural network layer of weights
class Layer(object):
    W = None
    _last = (None, None) # input, output

    def __init__(self, n, m):
        self.W = np.random.randn(m, n + 1) * 0.01

    # forward pass is the same as before.
    def forward(self, x):
        x = np.append(x, 1.) # add the fixed input for bias
        z = np.dot(self.W, x) # derivate: x
        y = 1. / (1. + np.exp(-z)) # derivate: y(1 - y)

        self._last = x, y
        return y

    # backward pass - compute the derivatives of each weight in this layer and
    # update them.
    def backward(self, dy):
        x, y = self._last

        # how the weights affect total loss (derivative w.r.t w)
        dz = dy * (y * (1 - y))
        dw = np.array([d * x for d in dz])

        # how the input (out of previous layer) affect total loss (derivative
        # w.r.t x). Derivates of the reverse of the forward pass.
        dx = np.dot(dz, self.W)
        dx = np.delete(dx, -1) # remove the bias input derivative

        # update
        # self.W -= ALPHA * dw
        return dw, dx

inp = OneHot(list(inp_vocab))
out = OneHot(list(out_vocab))
l1 = Layer(inp.N, H)
l2 = Layer(H, out.N)

w = np.zeros((out.N, 1 + inp.N)) # +1 for bias
last_l = float('inf')
for i in xrange(EPOCHS):
    np.random.shuffle(data)
    l = 0

    accuracy = 0
    for j in xrange(0, len(data), BATCHSIZE):
        minib = data[j:j+BATCHSIZE]
        dw1 = 0
        dw2 = 0
        for v, target in minib:
            x = inp.encode(*v) # encode the input features into multiple 1-of-key's
            h = l1.forward(x)
            y = l2.forward(h)

            # x = np.insert(x, 0, 1.) # fixed bias
            # y = np.dot(w, x) # compute the prediction
            res = out.decode(y) # a string, either "S" or "D"
            accuracy += 1 if res == target else 0 # simple as that!

            # loss and derivatives
            t = out.encode(target)
            l += (y - t) ** 2 / 2
            dy = y - t

            dw2_, dh = l2.backward(dy)
            dw1_, dx = l1.backward(dh)
            dw1 += dw1_
            dw2 += dw2_
            # dw += np.array([dyi * x for dyi in dy]) # Mx(1 + N) derivatives

        dw1 /= len(minib)
        dw2 /= len(minib)
        l1.W += STEP * -dw1 # mini-batch update
        l2.W += STEP * -dw2 # mini-batch update

    l /= len(data)
    l = sum(l)
    print "%s: LOSS = %s (%s); ACCURACY = %d of %d" % (i, l, l - last_l, accuracy, len(data))

    # if l - last_l > 0:
    #     STEP *= 0.9999
    #     print "%s: STEP = %f" % (i, STEP)
    # else:
    last_l = l

print
print "W = %s" % l1.W
