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

ALPHA = 0.5
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
data = [d for d in csv.DictReader(open("16_titanic.csv"))]
T = [[d.pop("Survived")] for d in data]
X = [d.items() for d in data]

# we have a lot of noise - if you try a batchsize of 1, you'll see that it takes
# a huge amount of time to converge. Other methods, like adapatable leanring
# rate can also work around that issue, arguably in a more generic and robust
# way.
BATCH = len(X) #  / 4

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
class TanH(object):
    W = None
    _last = (None, None) # input, output

    def __init__(self, n, m):
        self.N, self.M = n, m
        self.W = np.random.randn(m, n + 1) * 0.01

    # forward pass
    def forward(self, xs):
        bias = np.ones((xs.shape[0], 1))
        xs = np.concatenate((xs, bias), axis=1)
        ys = np.zeros((len(xs), self.M))
        for i, x in enumerate(xs):
            z = np.dot(self.W, x)
            y = np.tanh(z)
            ys[i] = y

        self._last = xs, ys
        return ys

    # backward pass
    def backward(self, dys):
        xs, ys = self._last
        dxs = np.zeros((len(dys), self.N))
        dws = np.zeros((len(dys),) + self.W.shape)
        for i, dy in enumerate(dys):
            x = xs[i]
            y = ys[i]

            # how the weights affect total error (derivative w.r.t w)
            dz = dy * (1 - y ** 2)
            dw = np.array([d * x for d in dz])
            dws[i] = dw

            # how the input (out of previous layer) affect total error
            # (derivative w.r.t x). Derivates of the reverse of the forward pass
            dx = np.dot(dz, self.W)
            dx = np.delete(dx, -1) # remove the bias input derivative
            dxs[i] = dx

        # update
        dw = sum(dws) / len(dws) # average out the weight derivatives
        self.W -= ALPHA * dw
        return dxs

class SquaredError(object):
    def forward(self, xs):
        self._ys = xs
        return xs

    def error(self, ts):
        ys = self._ys
        return (ys - ts) ** 2 / 2

    # squared error function just returns the simple derivative
    def backward(self, ts):
        ys = self._ys
        return ys - ts

class Softmax(object):
    def __init__(self, n, m):
        self.M, self.N = m, n
        self.W = np.random.randn(m, n + 1)

    def forward(self, xs):
        bias = np.ones((xs.shape[0], 1))
        xs = np.concatenate((xs, bias), axis=1)
        ys = np.zeros((len(xs), self.M))
        for i, x in enumerate(xs):
            z = np.dot(self.W, x)
            exps = np.exp(z - np.max(z))
            y = exps / np.sum(exps)
            ys[i] = y

        self._last = xs, ys
        return ys

    def error(self, ts):
        xs, ys = self._last
        es = np.zeros(ts.shape)
        for i, y in enumerate(ys):
            t = ts[i]
            es[i] = -np.log(y[np.argmax(t)])

        return es

    def backward(self, ts):
        xs, ys = self._last
        dxs = np.zeros((len(ts), self.N))
        dws = np.zeros((len(ts),) + self.W.shape)
        for i in range(len(ts)):
            x = xs[i]
            y = ys[i]
            t = ts[i]

            dy = y - t
            dw = np.array([d * x for d in dy])
            dws[i] = dw

            dx = np.dot(dy, self.W)
            dx = np.delete(dx, -1)
            dxs[i] = dx

        dw = sum(dws) / len(dws)
        self.W -= ALPHA * dw
        return dxs

# enode all of the inputs and targets
X = OneHot(INPUTS).encode(X)
T = OneHot(OUTPUTS).encode(T)

# create the layers
l1 = TanH(len(INPUTS), H)
l2 = TanH(H, len(OUTPUTS))
l3 = SquaredError()
layers = [l1, l2, l3]

indices = range(len(X))
last_e = float('inf')
for i in xrange(EPOCHS):
    np.random.shuffle(indices)

    e = 0
    accuracy = 0
    for j in xrange(0, len(indices), BATCH):
        minib = indices[j:j+BATCH]
        xs = X[minib]
        ts = T[minib]

        # forward
        ys = reduce(lambda xs, l: l.forward(xs), layers, xs)

        # backward
        e += sum(layers[-1].error(ts))
        dx = reduce(lambda ds, l: l.backward(ds), reversed(layers), ts)

        # calculate accuracy
        accuracy += sum(np.argmax(ys, axis=1) == np.argmax(ts, axis=1))

    e = sum(e) / len(indices)
    print "%s: ERROR = %s (%s); ACCURACY = %d of %d" % (i, e, e - last_e, accuracy, len(indices))

    last_e = e

print
print "W = %s" % l1.W
