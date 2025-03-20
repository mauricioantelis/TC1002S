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

print(dataset)

#there is no useless information

plt.scatter(dataset.Petal_Length, dataset.Petal_Width)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.show()

plt.scatter(dataset.Petal_Length, dataset.Sepal_Length)
plt.xlabel('Petal Length')
plt.ylabel('Sepal Length')
plt.show()

plt.scatter(dataset.Petal_Length, dataset.Sepal_Width)
plt.xlabel('Petal Length')
plt.ylabel('Sepal Width')
plt.show()

plt.scatter(dataset.Petal_Width, dataset.Sepal_Length)
plt.xlabel('Petal Width')
plt.ylabel('Sepal Length')
plt.show()

plt.scatter(dataset.Petal_Width, dataset.Sepal_Width)
plt.xlabel('Petal Width')
plt.ylabel('Sepal Width')
plt.show()

plt.scatter(dataset.Sepal_Length, dataset.Sepal_Width)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()

g = sns.pairplot(dataset[['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']], corner=True, diag_kind="kde")
plt.show()

g = sns.pairplot(dataset, corner=True, diag_kind="kde", hue='Flower')
plt.show()

# Get dataframes for each real cluster
df1 = dataset[dataset.Flower==0]
df2 = dataset[dataset.Flower==1]
df3 = dataset[dataset.Flower==2]

# Scatter plot of each real cluster for Petal
plt.scatter(df1.Petal_Length, df1.Petal_Width, label='Flower type 0', c='r', marker='o', s=64, alpha=0.3)
plt.scatter(df2.Petal_Length, df2.Petal_Width, label='Flower type 1', c='g', marker='o', s=64, alpha=0.3)
plt.scatter(df3.Petal_Length, df3.Petal_Width, label='Flower type 2', c='b', marker='o', s=64, alpha=0.3)

plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.legend()
plt.show()

# Scatter plot of each real cluster for Sepal
plt.scatter(df1.Sepal_Length, df1.Sepal_Width, label='Flower type 0', c='r', marker='o', s=64, alpha=0.3)
plt.scatter(df2.Sepal_Length, df2.Sepal_Width, label='Flower type 1', c='g', marker='o', s=64, alpha=0.3)
plt.scatter(df3.Sepal_Length, df3.Sepal_Width, label='Flower type 2', c='b', marker='o', s=64, alpha=0.3)

plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()
plt.show()

# Select variables (one, two, three, four)
X  = dataset[["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"]].values
#X  = dataset[["Petal_Length", "Petal_Width"]].values
#X  = dataset[["Sepal_Length", "Sepal_Width"]].values

# Get the class of each observation
y  = dataset["Flower"].values

print(X.shape)

print(y.shape)

indi0 = y == 0
print(np.sum(indi0))

indi1 = y == 1
print(np.sum(indi1))

indi2 = y == 2
print(np.sum(indi2))

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(C=1e5)

classifier.fit(X,y)

# Get a new observation
xnew = np.array([[5.5, 3.5, 1.5, 0.5]])
#xnew = np.array([[5.5, 2.5, 3.5, 1.5]])
#xnew = np.array([[6.5, 3.5, 5.5, 2.5]])

ypred = classifier.predict(xnew)

print(ypred)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

classifier2 = LogisticRegression(C=1e5)

classifier2.fit(X_train,y_train)

ypred = classifier2.predict(X_test)

print(y_test[0:29])
print(ypred[0:29])

from sklearn.metrics import accuracy_score
acctotal = 100*accuracy_score(y_test, ypred)
print("Total accuracy is " + str(acctotal))

classlabel = 0
idx = y_test==classlabel
acc = 100*accuracy_score(y_test[idx],ypred[idx])
print("Total accuracy for class " + str(classlabel) + " is " + str(acc))

classlabel = 1
idx = y_test==classlabel
acc = 100*accuracy_score(y_test[idx],ypred[idx])
print("Total accuracy for class " + str(classlabel) + " is " + str(acc))

classlabel = 2
idx = y_test==classlabel
acc = 100*accuracy_score(y_test[idx],ypred[idx])
print("Total accuracy for class " + str(classlabel) + " is " + str(acc))

from sklearn.metrics import confusion_matrix

CM = confusion_matrix(y_test, ypred, normalize='true')

from sklearn.metrics import ConfusionMatrixDisplay

disp = ConfusionMatrixDisplay(confusion_matrix=CM, display_labels=classifier2.classes_)
disp.plot()
plt.show()

