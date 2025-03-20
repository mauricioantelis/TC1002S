# Import the packages that we will be using
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

print(dataset)

# Print dataset
print(dataset.head())

# Print dataset shape
print(dataset.shape)

# Print column names
print(dataset.columns)

# Drop na
dataset = dataset.dropna()

print(dataset.shape)

# Encoding the categorical column: {"Iris-setosa":0, "Iris-versicolor":1, "Iris-virginica":2}
dataset = dataset.replace({"Flower":  {"Iris-setosa":0, "Iris-versicolor":1, "Iris-virginica":2} })

#Visualize the dataset
print(dataset)

# # Drop out non necesary columns
#There is not, non necesary columns, we will use al the data for comaprison, but if I would drop columns, I would do it as follows
#dataset.drop(['Sepal_Length', 'Sepal_Width'],axis='columns',inplace=True)
#
# #Visualize the dataset
print(dataset)

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

#Actividad A2
#1) a
#Identify the name of each column
print(dataset.columns)

#1) b
#Identify the type of each column
print(dataset.dtypes)

#1) c
#Minimum, maximum, mean, average, median, standar deviation
print(dataset.describe())

#2) Are there missing data? If so, create a new dataset containing only the rows with the non-missing data
#No there is no missing data

#3) Create a new dataset containing only the petal width and length and the type of Flower
print(dataset.loc[:,["Petal_Width", "Petal_Length","Flower"]])

#4) Create a new dataset containing only the setal width and length and the type of Flower
print(dataset.loc[:,["Sepal_Width", "Sepal_Length","Flower"]])

#5) Create a new dataset containing the setal width and length and the type of Flower encoded as a categorical numerical column
#I have no need given that my table already has the flowers by number

#Actividad A3
#1) Plot the histograms for each of the four quantitative variables
plt.hist(dataset.Sepal_Length)
plt.xlabel('Value')
plt.ylabel('Sepal Length')
plt.show()

plt.hist(dataset.Sepal_Width)
plt.xlabel('Value')
plt.ylabel('Sepal Width')
plt.show()

plt.hist(dataset.Petal_Length)
plt.xlabel('Value')
plt.ylabel('Petal Length')
plt.show()

plt.hist(dataset.Petal_Width)
plt.xlabel('Value')
plt.ylabel('Petal Width')
plt.show()

#2) Plot the histograms for each of the quantitative variables
#The same as 1

#3) Plot the boxplots for each of the quantitative variables
plt.boxplot(dataset.Sepal_Length)
plt.title('Sepal Length')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.show()

plt.boxplot(dataset.Sepal_Width)
plt.title('Sepal Width')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.show()

plt.boxplot(dataset.Petal_Length)
plt.title('Petal Length')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.show()

plt.boxplot(dataset.Petal_Width)
plt.title('Petal Width')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.show()

#4) Plot the boxplots of the petal width grouped by type of flower
sns.boxplot(data=dataset, y='Petal_Width', hue='Flower', gap=.4)
plt.title('Petal Width')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.show()

#5) Plot the boxplots of the setal length grouped by type of flower
sns.boxplot(data=dataset, y='Sepal_Length', hue='Flower', gap=.4)
plt.title('Sepal Length')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.show()

#6) Provide a description (explaination from your observations) of each of the quantitative variables
#It appears to be that the flower "0" has the least width of the petal with diference and that the flower "1" and "2" 
#are much closer to one another however, the flower "2" has a greater petal widht
#In the sepal length the 3 type of flowers have less of diference but it is still clear that the flower "0" is the least long
#and the "2" the longest
#In conclusion the flower "0" is generally smaller (in comparison with "1" and "2"), and "2" is bigger (again in comparison)
