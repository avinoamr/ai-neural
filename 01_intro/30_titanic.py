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

STEP = 5
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
T = [d.pop("Survived") for d in data]
X = [d.items() for d in data]

# we have a lot of noise - if you try a batchsize of 1, you'll see that it takes
# a huge amount of time to converge. Other methods, like adapatable leanring
# rate can also work around that issue, arguably in a more generic and robust
# way.
BATCHSIZE = len(data) #  / 4

class OneHot(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def encode(self, vs):
        indices = [self.vocab.index(v) for v in sorted(vs)]
        x = np.zeros(self.N)
        x[indices] = 1.
        return x

    def encode_all(self, data):
        return np.array([self.encode(d) for d in data])

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

# enode all of the inputs and targets
X = OneHot(INPUTS).encode_all(X)
T = OneHot(OUTPUTS).encode_all(T)
data = zip(X, T)

# create the layers
l1 = Layer(len(INPUTS), H)
l2 = Layer(H, len(OUTPUTS))
last_l = float('inf')
for i in xrange(EPOCHS):
    np.random.shuffle(data)
    l = 0

    accuracy = 0
    for j in xrange(0, len(data), BATCHSIZE):
        minib = data[j:j+BATCHSIZE]
        dw1 = 0
        dw2 = 0
        for x, t in minib:
            h = l1.forward(x)
            y = l2.forward(h)

            # loss and derivatives
            l += (y - t) ** 2 / 2
            dy = y - t

            dw2_, dh = l2.backward(dy)
            dw1_, dx = l1.backward(dh)
            dw1 += dw1_
            dw2 += dw2_

            # check the accuracy
            correct = np.argmax(y) == np.argmax(t)
            accuracy += 1 if correct else 0

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
