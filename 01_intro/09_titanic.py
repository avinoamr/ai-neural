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

# read the data from the CSV file
data = [d for d in csv.DictReader(open("09_titanic.csv"))]
N = 21

# we have a lot of noise - if you try a batchsize of 1, you'll see that it takes
# a huge amount of time to converge. Other methods, like adapatable leanring
# rate can also work around that issue, arguably in a more generic and robust
# way.
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
    x = np.zeros(N - 1)
    for k, v in vocabs.items():
        idx = v[d[k]]
        x[idx] = 1.

    return x

w = np.zeros(N)
for i in xrange(EPOCHS):
    random.shuffle(data)
    l = 0

    accuracy = 0
    for j in xrange(0, len(data), BATCHSIZE):
        minib = data[j:j+BATCHSIZE]
        dw = 0
        for d in minib:
            x = encode(d) # encode the input features into multiple 1-of-key's
            x = np.insert(x, 0, 1.) # fixed bias
            y = sum(x * w) # compute the prediction
            t = float(d["Survived"]) # encode the target correct output

            l += (y - t) ** 2 / 2
            dw += (y - t) * x

            accuracy += 1 if round(np.clip(y, 0, 1)) == t else 0

        dw /= len(minib)
        w += STEP * -dw # mini-batch update

    print "%s: LOSS = %s; ACCURACY = %d of %d" % (i, l, accuracy, len(data))

print
print "W = %s" % w
