# There are several obvious downsides to auto-regressive models:
#
#   1. We need many input neurons to account for all of the previous outputs
#       that we care about. The input space can grow huge when we need to
#       account for many previous output time-steps.
#   2. This also mean that each timestep is learned separately because it uses
#       differnet neurons. Learning will take much longer.
#   2. We are limited by how far back we need to look. The decision needs to be
#       made ahead of time when designing the model, instead of being learned.
#   3. We are only only relying on the previous input and output, instead of
#       learning more complicated dynamics or state.
#
# To illustrate this limitation, in this exercise we'll train a model that
# performs binary addition (and by extension, any positive integer addition),
# which can be defined as:
#
#   x1, x2, Carry
#   ---------------------------
#   0,  0,  0   => 0 (Carry: 0)
#   0,  1,  0   => 1 (Carry: 0)
#   1,  0,  0   => 1 (Carry: 0)
#   1,  1,  0   => 0 (Carry: 1)
#   0,  0,  1   => 1 (Carry: 0)
#   0,  1,  1   => 0 (Carry: 1)
#   1,  0,  1   => 0 (Carry: 1)
#   1,  1,  1   => 1 (Carry: 1)
#
# In some sense it's like the auto-regressive model, except that we don't need
# to concatenate the full history of events, but only the Carry. Essentially
# that carry neuron along with the input is enough to produce the output (and
# the new bit to carry over to the next timestep). The carry is the "memory" or
# state of the network spanning over multiple timesteps.
#
# A Recurent Neural Network is a system that's able to learn some internal
# state (in this case, simply the carry bit), learn when and how the state
# should be modified and finally access the state to determine the new value.
# One way to think about it is like having two different models: one for
# learning the recurrent state, and the other for learning the actual results
# normally.
import numpy as np
np.random.seed(1)

EPOCHS = 100000
ALPHA = 0.05
H = 5

# X1 = "001.010.0100."
# X2 = "010.011.0101."
# T  = "011.101.1001."

# binary addition, can be defined, per digit as:
#

X = [
    [2, 4],
    [1, 1],
    [3, 5],
    [8, 2],
    [18, 2],
    [50, 11],
    [7, 1]
]

T = [x[0] + x[1] for x in X]

# convert the input numbers in a long string of binary digits. For example:
# NOTE that this code is just data preparation, and therefore is not directly
# the implementation of the ML network. It can be skipped.
def enc(X, T):
    Xs0, Xs1, Ts = "", "", ""
    for x, t in zip(X, T):
        points = [x[0], x[1], t]

        # binary encode the values. bin(6)[2:] = "0b110"[2:] = "110"
        points = map(lambda p: bin(p)[2:], points)

        # align the values to the same number of digits by prepending zeros
        ndigits = max(map(lambda p: len(p), points))
        points = map(lambda p: p.rjust(ndigits, "0"), points)

        # reverse the digits - we compute binary addition going from right to left
        points = map(lambda p: "".join(reversed(p)), points)

        # end-of-number marker, to indicate when a binary number ends and a new one
        # begins.
        points = map(lambda p: p + ".", points)

        # generate the new strings.
        Xs0 += points[0]
        Xs1 += points[1]
        Ts += points[2]

    X, T = [], []
    for i in range(len(Ts)):
        c0, c1, t0 = Xs0[i], Xs1[i], Ts[i]

        # one-hot encode
        x0 = np.eye(3)["01.".index(c0)]
        x1 = np.eye(3)["01.".index(c1)]
        x = np.concatenate((x0, x1))
        t = np.eye(3)["01.".index(t0)]

        X.append(x)
        T.append(t)

    X = np.array(X)
    T = np.array(T)
    return X, T

# X, T = enc(X, T)

class Recurrent(object):
    def __init__(self, n, m):
        self.Wx = np.random.randn(m, n + 1) * 0.01
        self.Wh = np.random.randn(m, m) * 0.01
        self._h = np.zeros(m)
        self._prevs = []

    def forward(self, x):
        h = self._h
        x = np.append(x, 1.)
        z = np.dot(self.Wx, x) + np.dot(self.Wh, h)
        y = np.tanh(z)

        self._prevs.append((x, h, y))
        self._prevs = self._prevs[-25:] # truncate.
        self._h = y
        return y

    def backward(self, dy, t = 1):
        if t > len(self._prevs):
            return

        x, h, y = self._prevs[-t]
        dz = (1 - y ** 2) * dy
        dwx = np.array([d * x for d in dz])
        dwh = np.array([d * h for d in dz])

        dx  = np.dot(dz, self.Wx)
        dh  = np.dot(dz, self.Wh)
        self.Wx += ALPHA * -dwx
        self.Wh += ALPHA * -dwh

        self.backward(dh, t + 1)
        return np.delete(dx, -1)

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


l1 = Recurrent(6, H)
l2 = Softmax(H, 3)
layers = [l1, l2]
# data = zip(X, T)
for i in xrange(EPOCHS):
    e = 0.
    accuracy = 0

    X, T = [], []
    for _ in range(10):
        x0, x1 = np.random.randint(10), np.random.randint(10)
        X.append([x0, x1])
        T.append(x0 + x1)

    X1, T1 = enc(X, T)
    data = zip(X1, T1)
    for x, t in data:

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
