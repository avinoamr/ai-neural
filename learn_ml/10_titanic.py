import csv
import random
import numpy as np

STEP = 3.2
random.seed(1)

data = [d for d in csv.DictReader(open("titanic.csv"))]
N = 21

vocabs = {
    "Fare": { "cheap": 1, "low": 2, "medium": 3, "high": 4 },
    "Embarked": { "S": 5, "C": 6, "Q": 7 },
    "Age": { "kid": 8, "young": 9, "adult": 10, "old": 11 },
    "Family": { "alone": 12, "small": 13, "medium": 14, "big": 15 },
    "Pclass": { "1": 16, "2": 17, "3": 18 },
    "Sex": { "male": 19, "female": 20 }
}

losses = []
w = np.zeros(N)
for i in xrange(4000):
    random.shuffle(data)
    l = 0

    if i == 1000:
        STEP /= 2
    elif i == 2000:
        STEP = 0.0001

    accuracy = 0.0
    for d in data:
        # encode the data into N input neurons
        x = np.zeros(N)
        x[0] = 1. # bias

        for k, v in vocabs.items():
            idx = v[d[k]]
            x[idx] = 1.

        # encode the output
        t = float(d["Survived"])

        # compute the result
        y = sum(x * w)
        l += ((y - t) ** 2) / len(data)

        # derivatives
        dw = (2 * (y - t) * x) / len(data)

        if y > 0.5 and t == 1:
            accuracy += 1
        elif y <= 0.5 and t == 0:
            accuracy += 1

    w += STEP * dw * -1

    if i % 100 == 0:
        print "%s: LOSS = %s; CORRECT = %s" % (i, l, accuracy)

    losses.append(l)
    if len(losses) == 5:
        diffs, losses = np.diff(losses), []
        print "DIFFS = %s" % diffs
        if np.allclose(diffs, 0):
            break


print "ACCURACY %s%% = %s of %s" % (accuracy / len(data) * 100, accuracy, len(data))
print w
