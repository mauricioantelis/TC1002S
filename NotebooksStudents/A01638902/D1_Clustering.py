# Import the packages that we will be using
import numpy as np                  # For array
import pandas as pd                 # For data handling
import seaborn as sns               # For advanced plotting
import matplotlib.pyplot as plt     # For showing plots

# Note: specific functions of the "sklearn" package will be imported when needed to show concepts easily

# Define the col names for the iris dataset
colnames = ["Sepal_Length", "Sepal_Width","Petal_Length","Petal_Width", "Flower"]

# Dataset url
#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
url = Ruta + "datasets/iris/iris.csv"

# Load the dataset from HHDD
dataset  = pd.read_csv(url, header = None, names = colnames )

dataset

print(dataset.shape)

dataset.head()

dataset.columns


dataset = dataset .dropna()

print(dataset.shape)

# Encoding the categorical column
dataset = dataset.replace({"Flower":  {"Iris-setosa":0, "Iris-versicolor":1, "Iris-virginica":2} })

#Visualize the dataset
dataset



# # Drop out non necesary columns
# dataset.drop(['Sepal_Length', 'Sepal_Width'],axis='columns',inplace=True)
#
# #Visualize the dataset
# dataset


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

plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()
plt.show()

# Import sklearn KMeans
from sklearn.cluster import KMeans

# Define number of clusters
km = KMeans(n_clusters=3, n_init="auto")

# Do K-means clustering (assing each point in the dataset to a cluster)
#FlowerPred = km.fit_predict(dataset[['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']])
#FlowerPred = km.fit_predict(dataset[['Sepal_Length','Sepal_Width']])
Cluster1 = km.fit_predict(dataset[['Petal_Length','Petal_Width']] )

# Print estimated cluster of each observations in the dataset
Cluster1

# Print real cluster of each observations in the dataset
dataset.Flower.values

# Manual pairing of the labels of the estimated clusters with the real ones
Cluster1Paired = np.choose(Cluster1, [2, 0, 1]).astype(int) # CHANGE USING THE ORDER THE LABEL ESTIMATED
Cluster1Paired

# Automatic pairing of the labels of the estimated clusters with the real ones: WORK IN PROGRESS

# Import library
# from sklearn.metrics.pairwise import pairwise_distances_argmin

# Centroides of the real clusters
#real_cluster_centers  =

# Centroides of the estimated clusters
#esti_cluster_centers  = km2.cluster_centers_

# Compute order for the estimated clusters
#order = pairwise_distances_argmin(real_cluster_centers, esti_cluster_centers )

# Get ordered estimated clusters
#esti_cluster_centers = esti_cluster_centers[order]

# Get paired labels
#real_cluster_labels = pairwise_distances_argmin(X, real_cluster_centers)
#esti_cluster_lables = pairwise_distances_argmin(X, esti_cluster_centers)


# Add a new column to the dataset with the cluster information
dataset['Cluster1'] = Cluster1Paired

dataset

# Print the existing labels/names of the estimated clusters (use the method unique)
dataset.Cluster1.unique()
km.cluster_centers_


km.inertia_

km.n_iter_

df1 = dataset[dataset.Cluster1==0]
df2 = dataset[dataset.Cluster1==1]
df3 = dataset[dataset.Cluster1==2]

# Scatter plot of each estimated cluster
plt.scatter(df1.Petal_Length, df1.Petal_Width, label='Cluster 0', c='r', marker='o', s=32, alpha=0.3)
plt.scatter(df2.Petal_Length, df2.Petal_Width, label='Cluster 1', c='g', marker='o', s=32, alpha=0.3)
plt.scatter(df3.Petal_Length, df3.Petal_Width, label='Cluster 2', c='b', marker='o', s=32, alpha=0.3)

plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], color='black', marker='*', label='Centroides', s=256)

plt.title('Petal')
plt.xlabel('Length')
plt.ylabel('Width')
plt.legend()
plt.show()

# Get dataframes for each real cluster
df1 = dataset[dataset.Flower==0]
df2 = dataset[dataset.Flower==1]
df3 = dataset[dataset.Flower==2]

# Scatter plot of each real cluster
plt.scatter(df1.Petal_Length, df1.Petal_Width, label='Flower type 0', c='white', edgecolor='r', marker='^', s=64, alpha=0.9)
plt.scatter(df2.Petal_Length, df2.Petal_Width, label='Flower type 1', c='white', edgecolor='g', marker='<', s=64, alpha=0.9)
plt.scatter(df3.Petal_Length, df3.Petal_Width, label='Flower type 2', c='white', edgecolor='b', marker='>', s=64, alpha=0.9)

# Get dataframes for each estimated cluster
df1 = dataset[dataset.Cluster1==0]
df2 = dataset[dataset.Cluster1==1]
df3 = dataset[dataset.Cluster1==2]

# Scatter plot of each estimated cluster
plt.scatter(df1.Petal_Length, df1.Petal_Width, label='Cluster 0',      c='white', edgecolor='r', marker='^', s=16, alpha=0.9)
plt.scatter(df2.Petal_Length, df2.Petal_Width, label='Cluster 1',      c='white', edgecolor='g', marker='<', s=16, alpha=0.9)
plt.scatter(df3.Petal_Length, df3.Petal_Width, label='Cluster 2',      c='white', edgecolor='b', marker='>', s=16, alpha=0.9)

plt.title('Petal')
plt.xlabel('Length')
plt.ylabel('Width')
plt.legend()

#plt.xlim(4,6)
#plt.ylim(1,2)

plt.show()

# Intialize a list to hold sum of squared error (sse)
sse = []

# Define values of k
k_rng = range(1,10)

# For each k
for k in k_rng:
    # Create model
    km = KMeans(n_clusters=k, n_init="auto")
    # Do K-means clustering
    km.fit_predict(dataset[['Petal_Length','Petal_Width']])
    # Save sse for each k
    sse.append(km.inertia_)


# Plot sse versus k
plt.plot(k_rng,sse)

plt.title('Elbow plot')
plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.show()


# A list holds the silhouette coefficients for each k
#silhouette_coefficients = []

# Notice you start at 2 clusters for silhouette coefficient
#for k in range(2, 11):
#    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
#    kmeans.fit(scaled_features)
#    score = silhouette_score(scaled_features, kmeans.labels_)
#    silhouette_coefficients.append(score)


"""

# Intialize a list to hold silhouette coefficients
silhouette_coefficients = []

# Define values of k
k_rng = range(1,10)

# Parametrs
kmeans_kwargs = {"init": "random","n_init": 10,"max_iter": 300,"random_state": 42}

# For each k
for k in k_rng:
    # Create model
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    # Do K-means clustering
    kmeans.fit_predict(df[['Petal_Length','Petal_Width']])
    # Compute silhouette coefficient
    score = silhouette_score(df[['Petal_Length','Petal_Width']], kmeans.labels_)
    # Save silhouette coefficient for each k
    silhouette_coefficients.append(score)
"""

"""
# Plot silhouette coefficient versus k
plt.plot(k_rng,silhouette_coefficients)

plt.title('Silhouette Coefficients')
plt.xlabel('K')
plt.ylabel('silhouette_coefficients')
plt.show()
"""

"""
# Import library
from sklearn.preprocessing import MinMaxScaler

# Initialize scaler
scaler = MinMaxScaler()
"""

"""
# Scale data
scaler.fit(df[['Petal_Length']])
df['Petal_Length_Scaled'] = scaler.transform(df[['Petal_Length']])

scaler.fit(df[['Petal_Width']])
df['Petal_Width_Scaled'] = scaler.transform(df[['Petal_Width']])

df
"""

"""
# Scatter plot of the scaled data
plt.scatter(dataset.Petal_Length,dataset.Petal_Width)
plt.title('Petal')
plt.xlabel('Length')
plt.ylabel('Width')
plt.show()
"""

"""
# Initialize model and define number of clusters
km = KMeans(n_clusters=3)

# Do K-means clustering (assing each point in the dataset to a cluster)
#yp = km.fit_predict(dataset)
yp = km.fit_predict(dataset[['Petal_Length','Petal_Width']])

# Print estimated cluster of each point in the dataser
yp
"""

"""
# Add a new column to the dataset with the cluster information
dataset['Cluster2'] = yp

dataset
"""

"""
df1 = dataset[dataset.Cluster2==0]
df2 = dataset[dataset.Cluster2==1]
df3 = dataset[dataset.Cluster2==2]

plt.scatter(df1.Petal_Length, df1.Petal_Width, Label='Cluster 0')
plt.scatter(df2.Petal_Length, df2.Petal_Width, Label='Cluster 1')
plt.scatter(df3.Petal_Length, df3.Petal_Width, Label='Cluster 2')

plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], color='magenta', marker='*', label='Centroides', s=256)

plt.title('Petal')
plt.xlabel('Length')
plt.ylabel('Width')
plt.legend()
plt.show()
"""

#Remove all observations from one of the classes

# Discard observation for one of the classes, e.g., class "setosa": to have only two classes in our dataset
#Label2Remove = 3 # (1,2,3)
#df = df[df.Flower!=Label2Remove]
#df
