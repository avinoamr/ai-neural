import csv
import random
import numpy as np

STEP = 0.1
EPOCHS = 2000

random.seed(1)
data = [d for d in csv.DictReader(open("titanic.csv"))]
data, validation = data[:595], data[595:]
N = 21

vocabs = {
    "Fare": { "cheap": 1, "low": 2, "medium": 3, "high": 4 },
    "Embarked": { "S": 5, "C": 6, "Q": 7 },
    "Age": { "kid": 8, "young": 9, "adult": 10, "old": 11 },
    "Family": { "alone": 12, "small": 13, "medium": 14, "big": 15 },
    "Pclass": { "1": 16, "2": 17, "3": 18 },
    "Sex": { "male": 19, "female": 20 }
}

# encode the data into N input neurons
def encode(d):
    x = np.zeros(N)
    x[0] = 1. # bias

    for k, v in vocabs.items():
        idx = v[d[k]]
        x[idx] = 1.

    return x

w = np.zeros(N)
for i in xrange(EPOCHS):
    random.shuffle(data)
    l = 0

    accuracy = 0.0
    remaining = data
    while len(remaining) > 0:
        minib, remaining = remaining[:200], remaining[200:]
        dw = 0
        for d in minib:
            x = encode(d) # encode the input features into multiple 1-of-key's
            y = sum(x * w) # compute the prediction
            t = float(d["Survived"]) # encode the target correct output
            accuracy += 1 if round(y) == t else 0

            l += ((y - t) ** 2) / len(data) # compute the loss
            dw += (2 * (y - t) * x) / len(minib) # derivatives of the loss

        # mini-batch update
        w += STEP * dw * -1

    if i % 100 == 0:
        print "%s: LOSS = %s; CORRECT = %s" % (i, l, accuracy)

print "TRAINING %s%% = %s of %s" % (accuracy / len(data) * 100, accuracy, len(data))

accuracy = 0.0
for d in validation:
    x = encode(d) # encode the input features into multiple 1-of-key's
    y = sum(x * w) # compute the prediction
    t = float(d["Survived"]) # encode the target correct output
    accuracy += 1 if round(y) == t else 0

print "VALIDATION %s%% = %s of %s" % (accuracy / len(validation) * 100, accuracy, len(validation))
