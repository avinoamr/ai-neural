# Titanic data is a bit messy. This code contains the preprocessing and feature
# engineering required to tidy it up.
import re
import csv

r = csv.DictReader(open("titanic_original.csv"))
data = [d for d in r]
w = csv.DictWriter(open("titanic.csv", "w"), ["Fare", "Embarked", "Age",
    "FamilySize", "Pclass", "Sex", "Survived"])

w.writeheader()
for d in data:
    # fill in missing "Embarked". Only two missing, and based on their fare
    # price they're most likely to embark from C.
    if d["Embarked"] == "":
        d["Embarked"] = "C"

    # fill in missing "Age". By first estimating the age from the title in the
    # name (Miss, Mr, Master, etc.). But we have duplicate titles that mean the
    # same thing. So we first clean that up.
    title = re.search("([A-Za-z]+)\.", d["Name"]).group(0)
    title = { title: title, "Mlle.": "Ms.", "Mme.": "Ms.", "Ms.": "Ms.",
        "Miss.": "Ms.", "Dr.": "Mr.", "Major.": "Mr.", "Capt.": "Mr.",
        "Sir.": "Mr.", "Don.": "Mr.", "Lady.": "Mrs.", "Countess.": "Mrs.",
        "Jonkheer.": "Other.", "Col.": "Other.", "Rev.": "Other."
    }[title]

    # Now that the list of titles is de-duped, we can fill in the the mean age
    # by title
    if d["Age"] == "":
        d["Age"] = { "Mr.": "33", "Mrs.": "36", "Master.": "5", "Ms.": "22",
            "Other.": "46" }[title]

    # Turn "Age" into a discrete feature
    age = float(d["Age"])
    if age <= 16:
        d["Age"] = "kid"
    elif age <= 32:
        d["Age"] = "young"
    elif age <= 64:
        d["Age"] = "adult"
    else:
        d["Age"] = "old" #

    # Turn "Fare" into a discrete feature
    fare = float(d["Fare"])
    if fare <= 7.91:
        d["Fare"] = "cheap"
    elif fare <= 14.454:
        d["Fare"] = "low"
    elif fare <= 31.0:
        d["Fare"] = "medium"
    else:
        d["Fare"] = "high"

    # family size
    famsize = int(d["Parch"]) + int(d["SibSp"])
    if famsize == 0:
        d["FamilySize"] = "alone"
    elif famsize <= 3:
        d["FamilySize"] = "small"
    elif famsize <= 6:
        d["FamilySize"] = "medium"
    else:
        d["FamilySize"] = "big"

    # remove un-used
    del d["Name"]
    del d["Ticket"]
    del d["PassengerId"]
    del d["Parch"]
    del d["SibSp"]
    del d["Cabin"]

    w.writerow(d)
