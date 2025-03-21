import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

# Define the col names for the iris dataset
colnames = ["Sepal_Length", "Sepal_Width","Petal_Length","Petal_Width", "Flower"]

# Dataset url
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Load the dataset from HHDD
dataset  = pd.read_csv(url, header = None, names = colnames )

# Print dataset
print(dataset)
# Print dataset shape
print(dataset.shape)
# Print column names
print(dataset.columns)

# Drop na

print(dataset.dropna())

#Encoding the categorical column: {"Iris-setosa":0, "Iris-versicolor":1, "Iris-virginica":2}

dataset = dataset.replace({"Flower":  {"Iris-setosa":0, "Iris-versicolor":1, "Iris-virginica":2} })

#Visualize the dataset

print(dataset)

# Drop out non necesary columns

#No es necesario

# Scatter plot of Petal_Length vs Petal_Width
plt.scatter(dataset.Petal_Length,dataset.Petal_Width)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.show()


# Scatter plot of Petal_Length vs Sepal_Length
plt.scatter(dataset.Petal_Length,dataset.Sepal_Length)
plt.xlabel('Petal Length')
plt.ylabel('Sepal Length')
plt.show()

# Scatter plot of Petal_Length vs Sepal_Width
plt.scatter(dataset.Petal_Length,dataset.Sepal_Width)
plt.xlabel('Petal Length')
plt.ylabel('Sepal Width')
plt.show()

# Scatter plot of Petal_Width vs Sepal_Length
plt.scatter(dataset.Petal_Width,dataset.Sepal_Length)
plt.xlabel('Petal Width')
plt.ylabel('Sepal Length')
plt.show()

# Scatter plot of Petal_Width vs Sepal_Width
plt.scatter(dataset.Petal_Width,dataset.Sepal_Width)
plt.xlabel('Petal Width')
plt.ylabel('Sepal Width')
plt.show()


# Scatter plot of Sepal_Length vs Sepal_Width
plt.scatter(dataset.Sepal_Length,dataset.Sepal_Width)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()

# Pairplot: Scatterplot of all variables (not the flower type)
g = sns.pairplot(dataset[['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']], corner=True, diag_kind="kde")
#g.map_lower(sns.kdeplot, levels=4, color=".2")
plt.show()


# Pairplot: Scatterplot of all variables (not the flower type)
g = sns.pairplot(dataset, corner=True, diag_kind="kde", hue='Flower')
#g.map_lower(sns.kdeplot, levels=4, color=".2")
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

# Select variables (one, two, three, four)
X  = dataset[["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"]].values
#X  = dataset[["Petal_Length", "Petal_Width"]].values
#X  = dataset[["Sepal_Length", "Sepal_Width"]].values

# Get the class of each observation
y  = dataset["Flower"].values

# Understand the data X

print(X.shape)


# Understand the data y

print(y.shape)

# Calculate the number of observations in the dataset

print(np.sum(dataset["Flower"]))

# Calculate the number of observations for class 0

index = y == 0
print(np.sum(index))

# Calculate the number of observations for class 1

index = y == 1
print(np.sum(index))
# Calculate the number of observations for class 2
index = y == 2
print(np.sum(index))

# Import sklearn linear_model

from sklearn.linear_model import LogisticRegression 

# Initialize the classifier

slr = LogisticRegression(C=1e5)


# Fit the model to the training data

slr.fit(X,y)

# Get a new observation
# xnew = np.array([[5.5, 3.5, 1.5, 0.5]])
xnew = np.array([[5.5, 2.5, 3.5, 1.5]])
# xnew = np.array([[6.5, 3.5, 5.5, 2.5]])

# Print the new observation
print(xnew)

# Make the prediction using xnew
prediccion = slr.predict(xnew)


# Get the predicted class

print(prediccion)


# Import sklearn train_test_split

from sklearn.model_selection import train_test_split


# Split data in train and test sets

x_train, x_test, y_train, y_test = train_test_split(X,y, test_size= 0.20, random_state=30) 

# Number of observations in the train set
print(x_train.shape[0] + y_train.shape[0])

# Number of observations of each class in the train set
print(x_train.shape[0])
print(y_train.shape[0])
# Number of observations in the test set
print(x_test.shape[0] + y_test.shape[0])
# Number of observations of each class in the test set
print(x_test.shape[0])
print(y_test.shape[0])

# Initialize the classifier
classifier1 = LogisticRegression(C=1e5)
classifier = LogisticRegression()

# Fit the model to the training data

classifier1.fit(x_train, y_train)

# Make the predictions using the test set

prediccion = classifier1.predict(x_test)

# Explore real and predicted labels

print(y_test[0:29])
print(prediccion[0:29])

# Define a function to compute accuracy
def Accuracy(actual,prediccion):
    acc = np.sum(np.equal(actual, prediccion)) / len(actual)
    return acc
# Calculate total accuracy
acctoral = 100 * Accuracy(y_test, prediccion)
print(acctoral)
# Calculate total accuracy using sklearn.metrics
from sklearn.metrics import accuracy_score

acctotal = 100*accuracy_score(y_test, prediccion)
print(acctotal)


# Compute confussion matrix (normalized confusion matrix)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, prediccion)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
# Plot normalized confussion matrix
disp.plot(cmap="Blues")
plt.show()


#Activity

#Compare the accuracy of the classification using (a) the four variables, (b) the two Petal variables, and (c) the two Sepal variables. Which provides the best classification accuracy?



X = dataset[["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"]].values
Xpetal = dataset[["Petal_Length", "Petal_Width"]].values
Xsepal = dataset[["Sepal_Length", "Sepal_Width"]].values

y = dataset["Flower"].values

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

classifier2 = RandomForestClassifier()
classifier3 = KNeighborsClassifier()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=30)
classifier1.fit(x_train,y_train)
y_pred = classifier1.predict(X_test)
print(accuracy_score(y_test, y_pred) * 100)


# X_train, X_test, y_train, y_test = train_test_split(Xpetal, y, test_size=0.20, random_state=30)
# classifier.fit(x_test,y_test)
# y_pred = classifier1.predict(X_test)
# print(accuracy_score(y_test, y_pred) * 100)

# X_train, X_test, y_train, y_test = train_test_split(Xsepal, y, test_size=0.20, random_state=30)
# classifier.fit(x_test,y_test)
# y_pred = classifier1.predict(X_test)
# print(accuracy_score(y_test, y_pred) * 100)


#Using the four variables, try with two classifiers. Which provides the best performanc

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=30)
classifier2.fit(x_train,y_train)
y_pred = classifier2.predict(X_test)
print(accuracy_score(y_test, y_pred) * 100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=30)
classifier3.fit(x_train,y_train)
y_pred = classifier3.predict(X_test)
print(accuracy_score(y_test, y_pred) * 100)