# Importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Importing data
colnames = ["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width", "Flower"]
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
dataset = pd.read_csv(url, header=None, names=colnames)

# Understanding and preprocessing the data
# 1. Get a general 'feel' of the data
print(dataset.head())
print(dataset.shape)
print(dataset.columns)

# 2. Drop rows with any missing values
dataset = dataset.dropna()
print(dataset.shape)

# 3. Encoding the class label categorical column: from string to num
dataset = dataset.replace({"Flower": {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}})
print(dataset)

# 4. Discard columns that won't be used
# Uncomment the following lines if needed
# dataset.drop(['Sepal_Length', 'Sepal_Width'], axis='columns', inplace=True)
# print(dataset)

# 5. Scatter plot of the data
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

# Pairplot: Scatterplot of all variables
sns.pairplot(dataset[['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']], corner=True, diag_kind="kde")
plt.show()

sns.pairplot(dataset, corner=True, diag_kind="kde", hue='Flower')
plt.show()

# 6. Scatter plot of the data assigning each point to the cluster it belongs to
df1 = dataset[dataset.Flower == 0]
df2 = dataset[dataset.Flower == 1]
df3 = dataset[dataset.Flower == 2]

plt.scatter(df1.Petal_Length, df1.Petal_Width, label='Flower type 0', c='r', marker='o', s=64, alpha=0.3)
plt.scatter(df2.Petal_Length, df2.Petal_Width, label='Flower type 1', c='g', marker='o', s=64, alpha=0.3)
plt.scatter(df3.Petal_Length, df3.Petal_Width, label='Flower type 2', c='b', marker='o', s=64, alpha=0.3)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.legend()
plt.show()

plt.scatter(df1.Sepal_Length, df1.Sepal_Width, label='Flower type 0', c='r', marker='o', s=64, alpha=0.3)
plt.scatter(df2.Sepal_Length, df2.Sepal_Width, label='Flower type 1', c='g', marker='o', s=64, alpha=0.3)
plt.scatter(df3.Sepal_Length, df3.Sepal_Width, label='Flower type 2', c='b', marker='o', s=64, alpha=0.3)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()
plt.show()