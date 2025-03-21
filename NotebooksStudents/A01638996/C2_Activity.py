import numpy as np                  # For array
import pandas as pd                 # For data handling
import seaborn as sns               # For advanced plotting
import matplotlib.pyplot as plt     # For showing plots

# Define the col names for the iris dataset
colnames = ["Sepal_Length", "Sepal_Width","Petal_Length","Petal_Width", "Flower"]

# Dataset url
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Load the dataset from HHDD
dataset  = pd.read_csv(url, header = None, names = colnames )

print(dataset.head())

print(dataset.shape)

print(dataset.columns)

dataset = dataset.dropna()

dataset = dataset.replace({"Flower":  {"Iris-setosa":0, "Iris-versicolor":1, "Iris-virginica":2} })

#Compare the accuracy of the classification using (a) the four variables, (b) the two Petal variables, and (c) the two Sepal variables. Which provides the best classification accuracy?
#a)
X  = dataset[["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"]].values
y  = dataset["Flower"].values

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

classifier2 = LogisticRegression(C=1e5)

classifier2.fit(X_train,y_train)

ypred = classifier2.predict(X_test)

from sklearn.metrics import accuracy_score
acctotal = 100*accuracy_score(y_test, ypred)
print("Total accuracy using all the variables is: " + str(acctotal) + "%")

#b)
X  = dataset[["Petal_Length", "Petal_Width"]].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

classifier2 = LogisticRegression(C=1e5)

classifier2.fit(X_train,y_train)

ypred = classifier2.predict(X_test)

from sklearn.metrics import accuracy_score
acctotal = 100*accuracy_score(y_test, ypred)
print("Total accuracy using petal variables is: " + str(acctotal) + "%")

#c)
X  = dataset[["Sepal_Length", "Sepal_Width"]].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

classifier2 = LogisticRegression(C=1e5)

classifier2.fit(X_train,y_train)

ypred = classifier2.predict(X_test)

from sklearn.metrics import accuracy_score
acctotal = 100*accuracy_score(y_test, ypred)
print("Total accuracy using sepal variables is: " + str(acctotal) + "%")

#I repeated this code 5 times, 
#80% of the times the 4 variables (a) performed the best out of the three, but it always gave me more or equal to 90% 
#20% of the times the petal variable (b) performed the best, this also gave me always 90% or more, but (a) was better almost every time
#100% of the times the sepal variables (c) gave me the worst chance out of the three, the best succes rate was 83% 

print()

#Using the four variables, try with two classifiers. Which provides the best performance?
X  = dataset[["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"]].values
y  = dataset["Flower"].values

from sklearn.gaussian_process import GaussianProcessClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

classifier = LogisticRegression(C=1e5)

classifier.fit(X_train,y_train)

ypred = classifier.predict(X_test)

classifier2 = GaussianProcessClassifier().fit(X_train,y_train)

ypred2 = classifier2.predict(X_test)

from sklearn.metrics import accuracy_score
acctotal = 100*accuracy_score(y_test, ypred)
acctotal2 = 100*accuracy_score(y_test, ypred2)

print("Total accuracy using Logistic Regression: " + str(acctotal) + "%")
print("Total accuracy using Gaussian Process: " + str(acctotal2) + "%")

#In repeated this code 5 times
#40% of the times the percentage was the same, 93.33% both times
#20% the Logistic Regression performed the best, 100% against 93.33%
#40% the Gaussian Procces performed the best, 90% against 86.66%, and 96.66% against 86.66%

#Between both classifiers based on the 5 test, we could say that the Gaussian Process is better however,
#I believe that if we make more test we could find that either this initial statement is right or that both
#the classifiers are (almost) equally effective
