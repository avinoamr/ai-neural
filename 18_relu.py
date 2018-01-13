# Another activation function that has been widely adopted recently is the
# Rectified Linear Unit (ReLU) which finally removes the vanishing gradients
# problem by keeping most of the activation mostly linear:
#
#   relu(x) = max(0, x)
#
# As long as the input x is positive, the function will just return x, keeping
# the derivative constant and linear: 1.
import numpy as np
np.random.seed(1)

ALPHA = 0.01

X = np.eye(3)
T = np.eye(3)[np.arange(len(X)) % 3]

class ReLU(object):
    def __init__(self, n, m):
        self.W = np.random.random((m, n + 1)) * 0.1

    def forward(self, x):
        x = np.append(x, 1.)
        z = np.dot(self.W, x)
        y = np.maximum(0, z)
        self._last = x, y

        print "z= %s y = %s" % (z, y)
        return y

    def backward(self, dy):
        x, y = self._last
        dz = dy * (y > 0) * 1.
        dw = np.array([d * x for d in dz])
        dx = np.dot(dz, self.W)
        self.W -= ALPHA * dw
        return dw, np.delete(dx, -1)

class SquaredError(object):
    def forward(self, x):
        # squared error layer doesn't modify the output. We will see other error
        # functions that do modify the output (Softmax, for example).
        self._y = x
        return x

    def error(self, t):
        y = self._y
        return (y - t) ** 2 / 2

    # squared error function just returns the simple derivative
    def backward(self, t):
        y = self._y
        return [], y - t

l1 = ReLU(X.shape[1], T.shape[1])
l2 = SquaredError()

def gradients(layers, x, t, epsilon = 0.0001):
    # compute the error of the given x, t
    last = layers[len(layers) - 1]
    y = reduce(lambda x, l: l.forward(x), layers, x)
    e = last.error(t)
    print "e=", e

    # now, shift all of the weights in all of the layers, and for each such
    # weight recompute the error to determine how that weight affects it
    print
    dws = [] # output derivatives per layer
    for l in layers:
        # some layers may just do a calculation without any weights (like the
        # final error layers).
        w = getattr(l, "W", np.array([]))
        dw = np.zeros(w.shape) # output derivatives for the layer
        for i in range(len(w)):
            for j in range(len(w[i])):
                w[i][j] += epsilon # shift the weight by a tiny epsilon amount
                yij = reduce(lambda x, l: l.forward(x), layers, x)
                eij = last.error(t) # re-run the network for the new error
                print "e=", eij
                dw[i][j] = sum(e - eij) / epsilon # normalize the difference
                w[i][j] -= epsilon # rever our change

        dws.append(dw)

    return dws

for x, t in zip(X, T):
    y = l1.forward(x)
    y = l2.forward(y)
    print "x=", x
    print "y=", y
    print "t=", t

    e = l2.error(t)
    print "e=", e

    dw = gradients([l1, l2], x, t)
    print
    print "dw1=", dw[0]
    print "dw2=", dw[1]

    dw2, d = l2.backward(t)
    print "d=", d
    print "dw2=", dw2

    dw1, _ = l1.backward(d)
    print "dw1=", dw1

    diff = abs(dw1 - -dw[0])
    print np.all(diff < 0.0001)


    # dy = y - t
    # print "t=", t
    # print "e=", e


    # dw1 = gradients([l1], )
    # dw = l1.backward(dy)



    break


    print
