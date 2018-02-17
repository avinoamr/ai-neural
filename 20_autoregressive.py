# It has been established that deep neural networks can approximate any function
# down to any small error rate. But these functions are limited in scope: they
# only operate on a single input/output pair at a time. They maintain no state,
# and thus cannot be used to describe more complicated logic that may span
# multiple time steps. For example, numerical addition cannot be learned because
# it relies on the carry value per digit. Other examples include language
# modeling, translation, video capturing, etc.
import numpy as np
np.random.seed(1)

EPOCHS = 1500
ALPHA = 0.1
H = 10

# the data below is based on the example data from Udacity's Luis Serrano[1]:
# the input is the weather given in a singel day, either Sunny (s) or Rainy (r).
# the output is the type of food being cooked by a roomate at that day, either
# an Apple Pie (a), a Burger (b) or Chicken (c).
#
# We want to learn the rules that govern the decision of what's likely to be
# cooked as a function of the weather. But there's one caveat: on sunny days,
# the roomate prefers to enjoy the weather, rather than cook something new, so
# she will just use the leftovers from the previous day.
#
# A normal, non-recurrent network cannot learn this correlation, because its
# API is limited to a single input/output pair at a time. It has no knowledge
# of the previous day's output.
#
# [1] https://www.youtube.com/watch?v=UNmqTiOnRfg
X = "rrrsssrrsrsrsrrrssrsrsssrrsrsrrssrsrsrrsrrrssrssrrsrssrrsrrrrssrsr"
T = "abccccabbccaabcaaabbccccabbccabbbccaabccabcccaaabccaaabccabcaaabbc"

# encode X and T to 1-one vectors
X = np.array([np.eye(2)["rs".index(c)] for c in X])
T = np.array([np.eye(3)["abc".index(c)] for c in T])

class Linear(object):
    def __init__(self, n, m):
        self.W = np.random.randn(m, n + 1) * 0.01

    def forward(self, x):
        x = np.append(x, 1.) # add the fixed input for bias
        y = np.dot(self.W, x) # derivate: x
        self._x = x
        return y

    def backward(self, dy):
        x = self._x
        dw = np.array([d * x for d in dy])
        dx = np.dot(dy, self.W)
        self.W -= ALPHA * dw
        return np.delete(dx, -1)

class Sigmoid(Linear):
    def forward(self, x):
        z = super(Sigmoid, self).forward(x)
        y = 1. / (1. + np.exp(-z))
        self._y = y
        return y

    def backward(self, dy):
        y = self._y
        dz = dy * (y * (1 - y))
        return super(Sigmoid, self).backward(dz)

class Softmax(Linear):
    def forward(self, x):
        y = super(Softmax, self).forward(x)
        exps = np.exp(y - np.max(y))
        p = exps / np.sum(exps)
        self._p = p
        return p

    def error(self, t):
        p = self._p
        return -np.log(p[np.argmax(t)])

    def backward(self, t):
        p = self._p
        dy = p - t
        return super(Softmax, self).backward(dy)

# we will simply feed the previous day's output as part of the new input.
l1 = Sigmoid(5, H)
l2 = Softmax(H, 3)
layers = [l1, l2]

data = zip(X, T)
y = np.zeros(3) # initial previous value
for i in xrange(EPOCHS):
    # NOTE that we are not shuffling the data, because the order does matter!

    e = 0.
    accuracy = 0
    for x, t in data:

        # concatenate the previous output to the new input
        x = np.concatenate((y, x))

        # forward
        y = reduce(lambda x, l: l.forward(x), layers, x)

        # backward
        e += layers[-1].error(t)
        d = reduce(lambda d, l: l.backward(d), reversed(layers), t)

        # update the accuracy
        accuracy += 1 if np.argmax(t) == np.argmax(y) else 0

    e /= len(data)
    accuracy = accuracy * 100 / len(data)
    print "%s: ERROR = %s ; ACCURACY = %s%%" % (i, e, accuracy)
