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

plt.hist(dataset['x1'], bins=10, alpha=0.5, label='x1', color='blue', edgecolor='black')
plt.hist(dataset['x2'], bins=10, alpha=0.5, label='x2', color='green', edgecolor='black')
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.title('Histogramas de x1 y x2')
plt.show()


"""
Based on these plots, provide a description of your data:

Your response here: La grafica que muestra el histograma tiene representado el eje y como la frecuencia y el eje x como los valores de los
datos seleccionados, en el caso del segundo histograma se selecciono un color azul para la columna "x1" y un color verde para la columna "x2"

"""

#Plot the boxplot of one of the variables

plt.figure(figsize=(8, 5))
sns.boxplot(y=dataset['x1'])
plt.title('Boxplot de x1')
plt.ylabel('x1')
plt.show()

#Plot in the same figure the boxplot of two variables
plt.figure(figsize=(8, 6))
sns.boxplot(data=dataset[['x1', 'x2']])
plt.title('Boxplots de x1 y x2')
plt.ylabel('Valores')
plt.xticks(rotation=45)
plt.show()

"""
Based on these plots, provide a description of your data:

Your response here: el boxplot muestra un Mínimo (sin contar valores atípicos), muestra el 
Primer cuartil (Q1): El cual representa el 25% de los datos están por debajo de este valor, tambien se ve la 
Mediana (Q2) El valor central de los datos, el Tercer cuartil (Q3): Representa El 75% de los datos están por debajo de este valor.
Máximo (sin contar valores atípicos).
"""

#Plot the scatter plot between all pair of variables

sns.pairplot( dataset, diag_kind="kde", markers="o", plot_kws={'alpha':0.5})
plt.show()

"""
### Questions

Based on the previos plots, provide a full description of yout dataset

Your response: En los histogramas se muestran formas de campana lo que significa una distribuicion normal, los boxplots muestran
alguno valores atipicos auqneu parecen estar distribuidos de valores simetricos

"""


#Do Kmeans clustering assuming a number of clusters accorging to your scatter plots

#Import sklearn KMeans
from sklearn.cluster import KMeans

# Define number of clusters
kmeans = KMeans(n_clusters=8, random_state=42)


#Add to your dataset a column with the estimated cluster to each data point
dataset['clstr'] = kmeans.fit_predict(dataset)

#Print the number associated to each cluster
print(dataset.head())


#Print the centroids
print(kmeans.cluster_centers_)

#Print the intertia metric

print(kmeans.inertia_)

#Plot a scatter plot of your data using different color for each cluster. Also plot the centroids

sns.pairplot(dataset,hue="clstr",palette="Set2",plot_kws={"alpha": 0.5})
plt.suptitle("clustering", y = 1.02)
plt.show()

#Elbow plot

#Compute the Elbow plot

# Intialize a list to hold sum of squared error (sse)
sse = []

# Define values of k
k_rng = range(1,10)

# For each k
for k in k_rng:
    # Create model
    km = KMeans(n_clusters=k, n_init="auto")
    # Do K-means clustering
    km.fit_predict(dataset)
    # Save sse for each k
    sse.append(km.inertia_)


# Plot sse versus k
plt.plot(k_rng,sse)

plt.title('Elbow plot')
plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.show()


"""
Part 2

## Do clustering using the "digits" dataset

"""

#1) Load the dataset from "sklearn.datasets"

from sklearn.datasets import load_digits

#2) Plot some of the observations (add in the title the label/digit of that obserbation)

image = load_digits().images
label = load_digits().target

plt.figure(figsize=(5,5))

plt.subplot(3,3,1)
plt.imshow(image[0],cmap="gray")
plt.title("Label 0")
plt.axis("off")


plt.subplot(3,3,2)
plt.imshow(image[1],cmap="gray")
plt.title("Label 1")
plt.axis("off")

plt.subplot(3,3,3)
plt.imshow(image[2],cmap="gray")
plt.title("Label 2")
plt.axis("off")

plt.subplot(3,3,4)
plt.imshow(image[3],cmap="gray")
plt.title("Label 3")
plt.axis("off")

plt.subplot(3,3,5)
plt.imshow(image[4],cmap="gray")
plt.title("Label 4")
plt.axis("off")

plt.subplot(3,3,6)
plt.imshow(image[5],cmap="gray")
plt.title("Label 5")
plt.axis("off")

plt.subplot(3,3,7)
plt.imshow(image[6],cmap="gray")
plt.title("Label 6")
plt.axis("off")

plt.subplot(3,3,8)
plt.imshow(image[7],cmap="gray")
plt.title("Label 7")
plt.axis("off")

plt.subplot(3,3,9)
plt.imshow(image[8],cmap="gray")
plt.title("Label 8")
plt.axis("off")


plt.tight_layout
plt.show()



"""

3) Do K means clustering in the following cases:

* KmeansAll: Using all 64 variables/pixels/features

* Kmeans1row: Using only the 8 variables/pixels/features from the firt row

* Kmeans4row: Using only the 8 variables/pixels/features from the fourth row

* Kmeans8row: Using only the 8 variables/pixels/
features from the eighth row


"""

from sklearn.cluster import KMeans

clusters = 10

# --- KMeansPrimeraFila: Usando solo los 8 píxeles de la primera fila ---
datos_primera_fila = image[:, 0, :]  # Solo la primera fila de cada imagen 8x8 (shape: 1797, 8)
kmeans_primera_fila = KMeans(n_clusters=clusters, random_state=42)
kmeans_primera_fila.fit(datos_primera_fila)
etiquetas_primera_fila = kmeans_primera_fila.labels_


# --- KMeansCuartaFila: Usando solo los 8 píxeles de la cuarta fila ---
datos_cuarta_fila = image[:, 3, :]  # Solo la cuarta fila de cada imagen 8x8 (shape: 1797, 8)
kmeans_cuarta_fila = KMeans(n_clusters=clusters, random_state=42)
kmeans_cuarta_fila.fit(datos_cuarta_fila)
etiquetas_cuarta_fila = kmeans_cuarta_fila.labels_


# --- KMeansOctavaFila: Usando solo los 8 píxeles de la octava fila ---
datos_octava_fila = image[:, 7, :]  # Solo la octava fila de cada imagen 8x8 (shape: 1797, 8)
kmeans_octava_fila = KMeans(n_clusters=clusters, random_state=42)
kmeans_octava_fila.fit(datos_octava_fila)
etiquetas_octava_fila = kmeans_octava_fila.labels_


# --- Visualización de Resultados ---
fig, ejes = plt.subplots(2, 2, figsize=(10, 10))

# Visualización KMeansPrimeraFila
ejes[0, 0].scatter(range(len(etiquetas_primera_fila)), etiquetas_primera_fila, c=etiquetas_primera_fila, cmap='viridis')
ejes[0, 0].set_title('KMeansPrimeraFila: Primera Fila')
ejes[0, 0].set_xlabel('Índice')
ejes[0, 0].set_ylabel('Cluster')

# Visualización KMeansCuartaFila
ejes[0, 1].scatter(range(len(etiquetas_cuarta_fila)), etiquetas_cuarta_fila, c=etiquetas_cuarta_fila, cmap='viridis')
ejes[0, 1].set_title('KMeansCuartaFila: Cuarta Fila')
ejes[0, 1].set_xlabel('Índice')
ejes[0, 1].set_ylabel('Cluster')

# Visualización KMeansOctavaFila
ejes[1, 0].scatter(range(len(etiquetas_octava_fila)), etiquetas_octava_fila, c=etiquetas_octava_fila, cmap='viridis')
ejes[1, 0].set_title('KMeansOctavaFila: Octava Fila')
ejes[1, 0].set_xlabel('Índice')
ejes[1, 0].set_ylabel('Cluster')

plt.tight_layout()
plt.show()




#Plot the elbow point

sse = []

# Define values of k
k_rng = range(1,10)

sse = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(load_digits().data)
    sse.append(kmeans.inertia_)
plt.plot(range(1, 10), sse)
plt.xticks(range(1, 10))
plt.xlabel("Clusters")
plt.ylabel("SSE")
plt.show()

#1) Load the dataset from "sklearn.datasets"

from sklearn.datasets import load_digits

#2) Plot some of the observations  (add in the title the label/digit of that obserbation)


image = load_digits().images
label = load_digits().target

plt.figure(figsize=(5,5))

plt.subplot(3,3,1)
plt.imshow(image[0],cmap="gray")
plt.title("Label 0")
plt.axis("off")


plt.subplot(3,3,2)
plt.imshow(image[1],cmap="gray")
plt.title("Label 1")
plt.axis("off")

plt.subplot(3,3,3)
plt.imshow(image[2],cmap="gray")
plt.title("Label 2")
plt.axis("off")

plt.subplot(3,3,4)
plt.imshow(image[3],cmap="gray")
plt.title("Label 3")
plt.axis("off")

plt.subplot(3,3,5)
plt.imshow(image[4],cmap="gray")
plt.title("Label 4")
plt.axis("off")

plt.subplot(3,3,6)
plt.imshow(image[5],cmap="gray")
plt.title("Label 5")
plt.axis("off")

plt.subplot(3,3,7)
plt.imshow(image[6],cmap="gray")
plt.title("Label 6")
plt.axis("off")

plt.subplot(3,3,8)
plt.imshow(image[7],cmap="gray")
plt.title("Label 7")
plt.axis("off")

plt.subplot(3,3,9)
plt.imshow(image[8],cmap="gray")
plt.title("Label 8")
plt.axis("off")


plt.tight_layout
plt.show()


#3) Split the dataset in train and test

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
df = load_digits()
# Split data in train and test sets


x_train, x_test, y_train, y_test = train_test_split(df.data,df.target, test_size= 0.20, random_state=30) 


from sklearn.ensemble import RandomForestClassifier
classifierall = RandomForestClassifier()
classifierall.fit(x_train, y_train)
classifier1col = RandomForestClassifier()
classifier1col.fit(x_train[:, :1], y_train)
classifier4col = RandomForestClassifier()
classifier4col.fit(x_train[:, 3:4], y_train)
classifier8col = RandomForestClassifier()
classifier8col.fit(x_train[:, 7:8], y_train)

y_pred_all = classifierall.predict(x_test)
y_pred_1col = classifier1col.predict(x_test[:, :1])
y_pred_4col = classifier4col.predict(x_test[:, 3:4])
y_pred_8col = classifier8col.predict(x_test[:, 7:8])


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm0 = confusion_matrix(y_test, y_pred_all)
disp = ConfusionMatrixDisplay(confusion_matrix=cm0)
disp.plot(cmap="Blues")
plt.show()

cm1 = confusion_matrix(y_test, y_pred_1col)
disp = ConfusionMatrixDisplay(confusion_matrix=cm1)
disp.plot(cmap="Reds")
plt.show()


cm2 = confusion_matrix(y_test, y_pred_4col)

disp = ConfusionMatrixDisplay(confusion_matrix=cm2)
disp.plot(cmap="Greens")
plt.show()

cm3 = confusion_matrix(y_test, y_pred_8col)
disp = ConfusionMatrixDisplay(confusion_matrix=cm3)
disp.plot(cmap="Purples")
plt.show()

digit=9
correcto = np.where((y_test == digit) & (y_pred_all == digit))[0]
incorrecto = np.where((y_test == digit) & (y_pred_all != digit))[0]
indices= np.concatenate([correcto[:3], incorrecto[:3]])
plt.figure(figsize=(12, 6))
for index, i in enumerate(indices):
    image = x_test[i]
    true_label = y_test[i]
    predicted_label = y_pred_all[i]

    plt.subplot(2, 3, index + 1)
    plt.imshow(np.reshape(image, (8, 8)), cmap=plt.cm.gray)
    plt.title(f"True: {true_label} Pred: {predicted_label}", fontsize=14)
    plt.axis("off")

plt.tight_layout()
plt.show()


"""
Escribe tu description del nivel de logro del siguiente criterio de la subcompetencia

**Interpreta interacciones**. Interpreta interacciones entre variables relevantes en un problema, como base para la construcción de modelos
bivariados basados en datos de un fenómeno investigado que le permita reproducir la respuesta del mismo.

Respuesta:  pude lograr interpretar de manera efectiva las interacciones entre las variables de un problema, utilice estos conocimientos para construir 
modelos bivariados adecuados que regresan o reflejan una informacion mas preciza, tambien desarrolle 
La capacidad para identificar y analizar relaciones clave entre variables, tanto en términos de correlación como de causalidad



Escribe tu description del nivel de logro del siguiente criterio de la subcompetencia

**Construcción de modelos**. Es capaz de construir modelos bivariados que expliquen el comportamiento de un fenómeno.

desarrolle una habilidad para construir modelos bivariados y que pude atender el problema  analizando
el comportamiento de un dataset. Este nivel de logro se refleja en mi capacidad para seleccionar las variables relevantes (Eliminar datos innesecarios etc), 
identificar las relaciones entre ellas, y aplicar machine learning con herramientas apropiadas para construir modelos predictivos de gran calidad

"""

