#Importing libraries

import numpy as np                  # For array
import pandas as pd                 # For data handling
import seaborn as sns               # For advanced plotting
import matplotlib.pyplot as plt     # For showing plots

Ruta = "../../NotebooksStudents/A01638902/"

url = Ruta + "A01638902_x.csv"

#Do clustering using your dataset

#load your dataSet

dataset = pd.read_csv(url)

#print the first 7 rows

print(dataset.head(7))

#Print the las 4 rows

print(dataset.tail(4))

#use shape method

print(dataset.shape)

#Use de column methods

print(dataset.columns)

#Use the dtypes method

print(dataset.dtypes)

#What is the meaning of rows and columns?

"""
Your responses here

1. Las filas representan las observaciones o registros del dataset

2. Las columnas representan las caracteristicas o variables de los datos

3. Cada celda en el conjunto de datos es la intersección de una fila

"""

#print the statistical summary of your columns
print(dataset["x1"].describe())
print(dataset["x2"].describe())
print(dataset["x3"].describe())
print(dataset["x4"].describe())
print(dataset["x5"].describe())
print(dataset["x6"].describe())
print(dataset["x7"].describe())

"""

1. What is the minumum and maximum values of each variable:
(info in printed values)

2. What is the mean and standar deviation of each variable:
(info in printed values)

50 debe ser un valor que describa "la mitad" de los datos, también conocido como mediana. 25, 75 es el límite del cuarto superior/inferior de los datos.
    
    
"""

#Rename de columns using capital letters


dataset = dataset.rename(columns={"x1":"X1","x2":"X2","x3":"X3","x4":"X4","x5":"X5","x6":"X6","x7":"X7"})
print(dataset.head())

#Rename the columns to their original names

dataset = dataset.rename(columns={"X1":"x1","X2":"x2","X3":"x3","X4":"x4","X5":"x5","X6":"x6","X7":"x7"})
print(dataset.head())

#Use two different alternatives to get one of the columns

print(dataset["x1"])
print(dataset.x1)

#Get a slice of your data set: second and thrid columns and rows from 62 to 72

subset = dataset.iloc[62:73, 1:3]
print(subset)

#For the second and thrid columns, calculate the number of null and not null values and verify that their sum equals the total number of rows

Numbernull = dataset.isnull().sum()  # Número de valores nulos

print(Numbernull)

dataset.drop(dataset.columns[-1], axis=1)

"""
### Questions

Based on the previos results, provide a full description of yout dataset

Your response:

El Dataset consta de 289 filas y 9 columnas, donde el numero de filas indica la cantidad de observaciones X las columnas que son la cantidad de variables,
en este caso, todas las observaciones son de tipo float.

"""

"""
Data visualization

"""

#Plot in the histogram of one of the variables
plt.hist(dataset['x1'], bins=10, color='blue', edgecolor='black')
plt.xlabel('Valores de x1')
plt.ylabel('Frecuencia')
plt.title('Histograma de x1')
plt.show()

#Plot in the same figure the histogram of two variables

plt.hist(dataset['x1'], bins=10, alpha=0.5, label='col2', color='blue', edgecolor='black')
plt.hist(dataset['x2'], bins=10, alpha=0.5, label='col3', color='green', edgecolor='black')
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.title('Histogramas de x1 y x2')
plt.show()