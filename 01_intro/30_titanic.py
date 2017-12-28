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

ALPHA = 5
EPOCHS = 25000000
H = 30
INPUTS = [
    ("Fare", "cheap"), ("Fare", "low"), ("Fare", "medium"), ("Fare", "high"),
    ("Family", "alone"), ("Family", "small"), ("Family", "medium"), ("Family", "big"),
    ("Embarked", "S"), ("Embarked", "C"), ("Embarked", "Q"),
    ("Age", "kid"), ("Age", "young"), ("Age", "adult"), ("Age", "old"),
    ("Pclass", "1"), ("Pclass", "2"), ("Pclass", "3"),
    ("Sex", "male"), ("Sex", "female")
]
OUTPUTS = ["0", "1"]

# read the data from the CSV file and break the data into an input and output
# sets, each is a list of (k,v) tuples
data = [d for d in csv.DictReader(open("30_titanic.csv"))]
T = [[d.pop("Survived")] for d in data]
X = [d.items() for d in data]

# we have a lot of noise - if you try a batchsize of 1, you'll see that it takes
# a huge amount of time to converge. Other methods, like adapatable leanring
# rate can also work around that issue, arguably in a more generic and robust
# way.
BATCHSIZE = len(X) #  / 4

# OneHot encodes data of arbitrary features into a list of one-hot neuron
# activations, each either a zero or one.
class OneHot(list):
    def encode(self, data):
        x = np.zeros((len(data), len(self)))
        for i, vs in enumerate(data):
            indices = [self.index(v) for v in sorted(vs)]
            x[i][indices] = 1.
        return x

# Layer represents a single neural network layer of weights
class Layer(object):
    W = None
    _last = (None, None) # input, output

    def __init__(self, n, m):
        self.N, self.M = n, m
        self.W = np.random.randn(m, n + 1) * 0.01

    # forward pass
    # now accepts a list of BATCH inputs, xs - one per test case - each with N
    # values - one per input neuron. Computes a list of BATCH outputs, ys.
    def forward(self, xs):

        # add the bias in one go to all of the data by creating a new vector of
        # 1s and then concatenating it with the input xs.
        bias = np.ones((xs.shape[0], 1)) # can be a constant
        xs = np.concatenate((xs, bias), axis=1)

        # BATCH x M outputs
        ys = np.zeros((len(xs), self.M))
        for i, x in enumerate(xs):
            z = np.dot(self.W, x) # derivate: x
            y = 1. / (1. + np.exp(-z)) # derivate: y(1 - y)
            ys[i] = y

        self._last = xs, ys
        return ys

    # backward pass
    # now accepts a list of BATCH dys - one per test case - each with M values -
    # one per output neuron. Computes the average dw for all of these cases and
    # updates once for that average. It also returns a list of BATCH dxs for
    # backprop.
    def backward(self, dys):
        xs, ys = self._last
        dxs = np.zeros((len(dys), self.N)) # BATCH x N input derivatives
        dws = np.zeros((len(dys),) + self.W.shape) # BATCH x N+1 weight derivatives
        for i, dy in enumerate(dys):
            x = xs[i]
            y = ys[i]

            # how the weights affect total loss (derivative w.r.t w)
            dz = dy * (y * (1 - y))
            dw = np.array([d * x for d in dz])
            dws[i] = dw

            # how the input (out of previous layer) affect total loss (derivative
            # w.r.t x). Derivates of the reverse of the forward pass.
            dx = np.dot(dz, self.W)
            dx = np.delete(dx, -1) # remove the bias input derivative
            dxs[i] = dx

        # update
        dw = sum(dws) / len(dws) # average out the weight derivatives
        self.W -= ALPHA * dw
        return dxs

# enode all of the inputs and targets
X = OneHot(INPUTS).encode(X)
T = OneHot(OUTPUTS).encode(T)

# create the layers
l1 = Layer(len(INPUTS), H)
l2 = Layer(H, len(OUTPUTS))
indices = range(len(X))

last_l = float('inf')
for i in xrange(EPOCHS):
    np.random.shuffle(indices)
    l = 0

    accuracy = 0
    for j in xrange(0, len(indices), BATCHSIZE):
        minib = indices[j:j+BATCHSIZE]
        xs = X[minib]
        ts = T[minib]

        # forward
        hs = l1.forward(xs)
        ys = l2.forward(hs)

        # backward
        l += sum((ys - ts) ** 2 / 2)
        dys = ys - ts
        dhs = l2.backward(dys)
        dxs = l1.backward(dhs)

        # calculate accuracy
        for i in range(len(minib)):
            y = ys[i]
            t = ts[i]
            accuracy += 1 if np.argmax(y) == np.argmax(t) else 0

    l /= len(indices)
    l = sum(l)
    print "%s: LOSS = %s (%s); ACCURACY = %d of %d" % (i, l, l - last_l, accuracy, len(indices))

    last_l = l

print
print "W = %s" % l1.W
