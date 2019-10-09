import pandas as pd
import numpy as np

df = pd.read_csv("titanic_train.csv", low_memory=False)

# process NA with median
df.Age = df.Age.fillna(df.Age.median())

# change Factor values into bool or (0,1)
df.Sex[df.Sex == "male"] = 0
df.Sex[df.Sex == "female"] = 1

# process NA with mode
df.Embarked = df.Embarked.fillna("S")

df.Embarked[df.Embarked == "S"] = 0
df.Embarked[df.Embarked == "C"] = 1
df.Embarked[df.Embarked == "Q"] = 2

# ---------------Use Linear regression--------------------
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

predictions = []
lr = LinearRegression()
kf = KFold(n_splits=3, random_state=20)
df.reset_index()

for train_index, test_index in kf.split(df.index):
    train_X = df[features].loc[train_index]
    train_Y = df["Survived"].loc[train_index]
    lr.fit(train_X.values, train_Y.values)
    predict = lr.predict(df[features].loc[test_index])
    predictions.append(predict.ravel())

predictions = np.concatenate(predictions, axis=0)  # !!!
predictions[predictions >= 0.5] = 1
predictions[predictions < 0.5] = 0

accuracy = sum(predictions == df.Survived) / len(predictions)
print(accuracy)

# ------------Using Logistic Regression---------------
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression

# Initialize our algorithm
lr = LogisticRegression(random_state=1)
# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
scores = cross_validate(lr, df[features], df["Survived"], cv=3)
# Take the mean of the scores (because we have one for each fold)
print(scores["test_score"].mean())   # 0.7878787878787877

# ------------------------Random Forest-----------------------
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)
# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
rfc = cross_validate(RFC, df[features], df["Survived"], cv=5)
scores = rfc["test_score"]
print(scores.mean())  # scores.mean = 0.8193878828436333

# the result is not good as expected,we need to adjust parameters
RFC = RandomForestClassifier(random_state=1, n_estimators=100, min_samples_split=4, min_samples_leaf=2)
rfc = cross_validate(RFC, df[features], df["Survived"], cv=5)
scores = rfc["test_score"]
print(scores.mean())  # scores.mean = 0.8294814821119466

# the result is still not nice, we could create some manual indexes
df["FamilySize"] = df["SibSp"] + df["Parch"]
df["NameLength"] = df["Name"].apply(lambda x: len(x))  # !!!!!!!! lambda!!!

import re

def get_title(name):
    # Use a regular expression to search for a title. Titles always consist of capital and lowercase letters, and end with a period.
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

# Get all the titles and print how often each one occurs.
titles = df["Name"].apply(lambda x:  re.search(' ([A-Za-z]+)\.', x).group(1))


# Map each title to an integer.  Some titles are very rare, and are compressed into the same codes as other titles.
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8,"Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
for k, v in title_mapping.items():
    titles[titles == k] = v

# Verify that we converted everything.
print(pd.value_counts(titles))

# Add in the title column.
df["Title"] = titles


# ---------------------------feature select ---------------------
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "Title", "NameLength"]

# Perform feature selection
selector = SelectKBest(f_classif, k=5)
selector.fit(df[features], df["Survived"])

# Get the raw p-values for each feature, and transform from p-values into scores
scores = -np.log10(selector.pvalues_)

# Plot the scores.  See how "Pclass", "Sex", "Title", and "Fare" are the best?
plt.bar(range(len(features)), scores)
plt.xticks(range(len(features)), features, rotation='vertical')
plt.show()

# Pick only the four best features.
predictors = ["Pclass", "Sex", "Fare", "Title"]
rfc = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=8, min_samples_leaf=4)
rfc.fit(df[predictors],df.Survived)
rfc.score(df[predictors],df.Survived)    # score = 0.8787878787878788


# --------------------ensemble to boost accuracy---------------
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

# We're using the more linear predictors for the logistic regression, and everything with the gradient boosting classifier.
algorithms = [[GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3),["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title"]],
              [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]]

kf = KFold(n_splits=3, random_state=1)

predictions = []
for train_index, test in kf.split(df.index):
    train_y = df["Survived"].iloc[train_index]
    full_test_predictions = []
    # Make predictions for each algorithm on each fold
    for alg, predictors in algorithms:
        alg.fit(df[predictors].iloc[train_index], train_y)
        # The .astype(float) is necessary to convert the dataframe to all floats and avoid an sklearn error.
        test_predictions = alg.predict_proba(df[predictors].iloc[test].astype(float))[:, 1]
        # ! why [:,1]? 1 means the probability of being 1(survived), 0 means the probability of being 0( died)

        full_test_predictions.append(test_predictions)
    # Use a simple ensembling scheme -- just average the predictions to get the final classification.
    test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2
    # Any value over .5 is assumed to be a 1 prediction, and below .5 is a 0 prediction.
    test_predictions[test_predictions <= .5] = 0
    test_predictions[test_predictions > .5] = 1
    predictions.append(test_predictions)

predictions = np.concatenate(predictions, axis=0)

accuracy = sum(predictions == df["Survived"]) / len(predictions)
print(accuracy)

# Now, we add the family size column.
predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title"]

algorithms = [[GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), predictors],
    [LogisticRegression(random_state=1), predictors]]

full_predictions = []
for alg, predictors in algorithms:
    # Fit the algorithm using the full training data.
    alg.fit(df[predictors], df["Survived"])
    # Predict using the test dataset.  We have to convert all the columns to floats to avoid an error.
    predictions = alg.predict_proba(df[predictors].astype(float))[:,1]
    full_predictions.append(predictions)

# The gradient boosting classifier generates better predictions, so we weight it higher.
predictions = (full_predictions[0] * 3 + full_predictions[1]) / 4
print(predictions)
predictions[predictions <= .5] = 0
predictions[predictions > .5] = 1
accuracy = sum(predictions == df["Survived"]) / len(predictions)
print(accuracy)
