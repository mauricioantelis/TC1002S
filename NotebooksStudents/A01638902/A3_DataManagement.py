import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



Ruta = "../../NotebooksStudents/A01638902"

# url = Ruta + "/cartwheel.csv"


df = pd.read_csv(url)


print(df.head())

age_counts = df["Age"].value_counts()
Gender_counts = df["Gender"].value_counts()

print(age_counts)
print(Gender_counts)

age_proportion = df['Age'].value_counts(normalize=True)
print(age_proportion)

# Total number of observations
print("Observaciones: ", df.size)


# total number of null observations in Age
print("Observaciones nulas de edad: ", df['Age'].isnull().sum())


# Total number of counts in Age (excluding missing values)
print("Observaciones de counts en edad", df['Age'].count())

# Plot histogram of Age
plt.figure(figsize=(10,6))
sns.histplot(df["Age"].dropna(), kde=True, bins=20, color="blue")
plt.title("Distribucion de edad")
plt.xlabel("Edad")
plt.ylabel("Frecuencia")
plt.show()

# Plot distribution of CWDistance

plt.figure(figsize=(10,6))
sns.histplot(df["CWDistance"].dropna(), kde=True, bins=20, color="blue")
plt.title("Distribucion de Distancia")
plt.xlabel("Edad")
plt.ylabel("Frecuencia")
plt.show()
# Plot histogram of both the Age and the Wingspan

plt.figure(figsize=(10,6))
sns.histplot(df["Age"].dropna(),kde="True",bins=20,color="blue",label="Age", alpha=0.5)
sns.histplot(df["Wingspan"].dropna(),kde=True,bins=20, color="red",label="Wingspan",alpha=0.5)
plt.title('Distribution of Age and Wingspan')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# Create the boxplot of the "CWDistance"
plt.figure(figsize=(8, 5))
sns.boxplot(y=df['CWDistance'])
plt.title('Boxplot de CWDistance')
plt.ylabel('CWDistance')
plt.show()
# Create the boxplot of the "Height"
plt.figure(figsize=(8, 5))
sns.boxplot(y=df['Height'])
plt.title('Boxplot de Height')
plt.ylabel('Height')
plt.show()

# Create the boxplots of the "CWDistance", "Height" and "Wingspan"
plt.figure(figsize=(8, 6))
sns.boxplot(data=df[['CWDistance', 'Height', 'Wingspan']])
plt.title('Boxplotsdef CWDistance, Height, y Wingspan')
plt.ylabel('Valores')
plt.xticks(rotation=45)
plt.show()


# Create side-by-side boxplots of the "CWDistance" grouped by "Gender"
sns.boxplot(data=df, y='CWDistance', hue='Gender', gap=.4)
plt.show()
# Create side-by-side boxplots of the "Glasses" grouped by "Gender"
sns.boxplot(data=df, x="Glasses", y="Height", hue="Gender")
plt.show()


# Create a boxplot and histogram of the "tips" grouped by "Gender"


# scatter plot between two variables
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='CWDistance', y='Height')
plt.title("CWDistance comparado con Height")
plt.xlabel('CWDistance')
plt.ylabel('Height')
plt.show()

# # scatter plot between two variables (one categorical)
plt.figure(figsize=(8, 6))
sns.stripplot(data=df, x='Gender', y='CWDistance', jitter=True)
plt.title('CWDistance y Gender')
plt.xlabel('Gender')
plt.ylabel('CWDistance')
plt.show()

# scatter plot between two variables (one categorical)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='CWDistance', y='Height', hue='Gender')
plt.title('CWDistance y Height agrupado por Gender')
plt.xlabel('CWDistance')
plt.ylabel('Height')
plt.legend(title='Gender')
plt.show()


# scatter plot between two variables grouped according to a categorical variable and with size of markers

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df, 
    x='CWDistance', 
    y='Height', 
    hue='Gender',    
    size='Wingspan', 
    sizes=(20, 200), 
    alpha=0.7       
)
plt.title("scatter plot between two variables grouped according to a categorical variable and with size of markers")
plt.xlabel('CWDistance')
plt.ylabel('Height')
plt.legend(title='Gender')
plt.show()

# Pairplot: Scatterplot of "Age","CWDistance","Height",'Wingspan'

sns.pairplot(df[['Age', 'CWDistance', 'Height', 'Wingspan']], diag_kind='kde')
plt.show()

#Activity: 

urlIris = Ruta + "/iris.csv"

dfIris = pd.read_csv(urlIris)


# Repeat this tutorial with the iris data set and respond to the following inquiries



# Plot the histograms for each of the four quantitative variables (sepal.length, sepal.width, petal.length, petal.width)

#sepal.length
plt.figure(figsize=(10,6))
sns.histplot(dfIris["sepal.length"].dropna(), kde=True, bins=20, color="blue")
plt.title("Histograma de largo de Sepalo")
plt.xlabel("Medidas")
plt.ylabel("Cantidad de Datos")
plt.show()

#Sepal.width
plt.figure(figsize=(10,6))
sns.histplot(dfIris["sepal.width"].dropna(), kde=True, bins=20, color="blue")
plt.title("Histograma de ancho de Sepalo")
plt.xlabel("Medidas")
plt.ylabel("Cantidad de Datos")
plt.show()

#petal.length

plt.figure(figsize=(10,6))
sns.histplot(dfIris["petal.length"].dropna(), kde=True, bins=20, color="blue")
plt.title("Histograma de largo de petalo")
plt.xlabel("Medidas")
plt.ylabel("Cantidad de Datos")
plt.show()

#Petal.width

plt.figure(figsize=(10,6))
sns.histplot(dfIris["petal.width"].dropna(), kde=True, bins=20, color="blue")
plt.title("Histograma de ancho de petalo")
plt.xlabel("Medidas")
plt.ylabel("Cantidad de Datos")
plt.show()



# Plot the boxplots for each of the quantitative variables

plt.figure(figsize=(8, 5))
sns.boxplot(y=dfIris['sepal.length'])
plt.title('Boxplot de sepal.length')
plt.ylabel('Medida')
plt.show()


plt.figure(figsize=(8, 5))
sns.boxplot(y=dfIris['sepal.width'])
plt.title('Boxplot de sepal.width')
plt.ylabel('Medida')
plt.show()


plt.figure(figsize=(8, 5))
sns.boxplot(y=dfIris['petal.length'])
plt.title('Boxplot de petal.length')
plt.ylabel('Medida')
plt.show()


plt.figure(figsize=(8, 5))
sns.boxplot(y=dfIris['petal.width'])
plt.title('Boxplot de petal.width')
plt.ylabel('Medida')
plt.show()


# Plot the boxplots of the petal width grouped by type of flower

sns.boxplot(data=dfIris, y='petal.width', hue='variety', gap=.4) #En mi dataset la variable "variety es el tipo de flor"
plt.show()


# Plot the boxplots of the setal length grouped by type of flower
sns.boxplot(data=dfIris, y='sepal.length', hue='variety', gap=.4) 
plt.show()

