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
import random
import numpy as np
random.seed(1)

STEP = 0.1
EPOCHS = 100

# read the data from the CSV file and break the data into an input and output
# sets, each is a list of (k,v) tuples
data = [d for d in csv.DictReader(open("30_titanic.csv"))]

N = 20

vocabs = [
    ("Fare", "cheap"), ("Fare", "low"), ("Fare", "medium"), ("Fare", "high"),
    ("Embarked", "S"), ("Embarked", "C"), ("Embarked", "Q"),
    ("Age", "kid"), ("Age", "young"), ("Age", "adult"), ("Age", "old"),
    ("Family", "alone"), ("Family", "small"), ("Family", "medium"), ("Family", "big"),
    ("Pclass", "1"), ("Pclass", "2"), ("Pclass", "3"),
    ("Sex", "male"), ("Sex", "female")
]

# we have a lot of noise - if you try a batchsize of 1, you'll see that it takes
# a huge amount of time to converge. Other methods, like adapatable leanring
# rate can also work around that issue, arguably in a more generic and robust
# way.
BATCHSIZE = len(data) / 4

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

inp = OneHot(vocabs)
out = OneHot([("Survived", "0"), ("Survived", "1")])
w = np.zeros((2, 1 + N)) # +1 for bias
for i in xrange(EPOCHS):
    random.shuffle(data)
    l = 0

    accuracy = 0
    for j in xrange(0, len(data), BATCHSIZE):
        minib = data[j:j+BATCHSIZE]
        dw = 0
        for d in minib:
            x = inp.encode(*d.items()) # encode the input features into multiple 1-of-key's
            x = np.insert(x, 0, 1.) # fixed bias
            y = np.dot(w, x) # compute the prediction
            res = out.decode(y) # a string, either "S" or "D"
            target = ("Survived", d["Survived"])
            accuracy += 1 if res == target else 0 # simple as that!

            # loss and derivatives
            t = out.encode(*d.items())
            l += (y - t) ** 2 / 2
            dy = y - t
            dw += np.array([dyi * x for dyi in dy]) # Mx(1 + N) derivatives

        dw /= len(minib)
        w += STEP * -dw # mini-batch update

    print "%s: LOSS = %s; ACCURACY = %d of %d" % (i, l, accuracy, len(data))

print
print "W = %s" % w
