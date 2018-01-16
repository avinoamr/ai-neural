# While playing around with the sigmoid activation functions, I've encountered
# two major problems:
#
#   (1) Vanishing Gradients. That's a well-known issue with sigmoid (and tanh)
#       functions. What happens is that if we start with small initial weights
#       (and even worse if we don't), the learned weights will eventually get
#       (absolutely) big as the model gains confidence in its assumptions. At
#       this point, the sigmoid function is almost flat (limiting at 1 or 0),
#       thus its derivative is almost zero. At this point almost no learning
#       takes place. Using small initial weights and making sure to randomize
#       the training data is usually enough to overcome this limitation.
#
#   (2) Slow hidden learning. This one, at the time of writing, was much less
#       documented. I barely managed to find references to it, causing me to
#       believe that I might've missed something critical in the reasoning (I
#       still do). What happens is that it took a very large learning rate to
#       eliminate the error, unless the dataset was big and diverse enough.
#       The reason was that the derivate of the last layer is multiplied by its
#       weights to become the derivative of the previous hidden layer. If these
#       weights are almost zero, the derivative - big as it may be - would be
#       multiplied by zero, causing an almost zero derivative on the hidden
#       layer. Think about it this way:
#
#           w1 ~= 0     ; w2 ~= 0
#           h  ~= sigmoid(w1 * x) = sigmoid(0 * x)   ~= 0.5
#           y  ~= sigmoid(w2 * h) = sigmoid(0 * 0.5) ~= 0.5
#
#       And indeed, i've seen that the output y was almost always closely around
#       0.5. Going backwards:
#
#           derror      (y - t) <= 0.5
#           dsigmoid    (y * (1 - y)) <= 0.25
#           dy = derror * dsigmoid * x  <= 0.125
#
#       The largest derivative possible for the last layer is 0.125. But now,
#       the derivative to the hidden layer is:
#
#           dh ~= w1 * dy <= 0 * 0.125
#
#       That's a very small number. This is what happens during learning:
#
#       a. h is usually 0.5 regardless of its input, and learns very very
#           slowly. y is also usually 0.5, but learns more rapidly based on the
#           rather constant (0.5) input from h.
#       b. Over a few iterations, y learns to reduce the error using a very weak
#           correlation with the input, as it remains 0.5 while learning of h
#           is slow.
#       c. This will gradually and cause the w2 to grow, starting to make a
#           bigger impact on the previous layer which will start learning as
#           well.
#       d. As that happens, the final layer y will start seeing more diverse
#           inputs causing it to learn in a different direction.
#
#       Indeed that's what I've been seeing: error drops fast for the first few
#       iterations and then it slows to almost a halt (while the weights grow),
#       and then it will drop again as the hidden layers are beginning to learn
#       thus the input starts to have more of an impact on the final result.
#       This problem if further multiplied for every hidden layer.
#
#       NOTE that if we have enough data and epochs, this problem becomes less
#       significant, because once the hidden layer joins the party - everything
#       learns fast. There's basically a delayed-learning mechanism. Perhaps
#       that's why it wasn't very documented?
#
# Whoa. So there's a catch-22 at play: if we choose small weights, the final
# layer learns fastest, but the hidden layer (and thus the input) will barely
# make a dent. And vice-versa.
#
# It would be best if we can find an non-linear activation function where we can
# use relatively big initial weights (0.5?), while still maintaining a high
# initial derivative? Well, there's tanh which is a scaled version of sigmoid.
# It increases the derivate, thus mildly mitigating the problem.
#
# The newest kid on the block is the Rectified Linear Unit (ReLU) activation
# function. It's beautifully simple and arguably it's more similar to biological
# neural networks:
#
#   relu(z) = max(0, z)
#
# It just removes all negative values, and otherwise it's linear. It has a few
# important properties:
#
#   1. The break-point at z < 0 is enough non-linearity to approximate any
#       function.
#   2. Its derivative in the linear part (z > 0) is a constant 1, so weights can
#       be arbitrarily big (good for learning the hidden layers)
#   3. Its derivative in the flat part (z < 0) is a constant 0. So once it gets
#       there, there will be no learning. Known as the "Dying ReLU Problem" this
#       is the biggest known problem with ReLUs and is mitigated by a few
#       variants, like the Leaky ReLU [1]. This limitaiton makes it important to
#       use a small learning rate in order to prevent the model from going too
#       far in the negative direction early in the learning.
#
# [1] https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#Leaky_ReLUs
import numpy as np
np.random.seed(1)

ALPHA = 0.05        # back to a normal learning rate
EPOCHS = 500
H = 2

# we will re-implement the XOR example with ReLU.
X = np.array([ [0., 0.], [0., 1.], [1., 0.], [1., 1.] ])
T = np.array([ [0.],     [1.],     [1.],     [0.]     ])

class ReLU(object):
    def __init__(self, n, m):
        self.W = np.random.random((m, n + 1))

    def forward(self, x):
        x = np.append(x, 1.)
        z = np.dot(self.W, x)
        y = np.maximum(0, z)
        self._last = x, y
        return y

    def backward(self, dy):
        x, y = self._last
        dz = dy * (y > 0) * 1. # if y > 0: 1, otherwise: 0.
        dw = np.array([d * x for d in dz])
        dx = np.dot(dz, self.W)
        self.W -= ALPHA * dw
        return np.delete(dx, -1)

class SquaredError(object):
    def forward(self, x):
        self._y = x
        return x

    def error(self, t):
        y = self._y
        return (y - t) ** 2 / 2

    def backward(self, t):
        y = self._y
        return y - t

data = zip(X, T)
l1 = ReLU(X.shape[1], H)
l2 = ReLU(H, T.shape[1])
l3 = SquaredError()
layers = [l1, l2, l3]
for i in xrange(EPOCHS):
    np.random.shuffle(data)
    e = 0.
    accuracy = 0
    for x, t in data:
        # forward
        y = reduce(lambda x, l: l.forward(x), layers, x)

        # backward
        e += l3.error(t)
        d = reduce(lambda d, l: l.backward(d), reversed(layers), t)

        # we're rounding again for accuracy calculation because I didn't want
        # to have multi-class inputs and outputs.
        accuracy += 1 if round(np.clip(y, 0, 1)) == t else 0

    e /= len(data)
    print "%s: ERROR = %s ; ACCURACY = %s" % (i, sum(e), accuracy)

print

for x, t in data:
    h = l1.forward(x)
    y = l2.forward(h)
    print "x=", x
    print "h=", h
    print "y=", y
    print "t=", t
    print
