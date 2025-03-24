import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Ruta del archivo CSV
ruta = './A07106692_X.csv'
df = pd.read_csv(ruta)

# Eliminar la columna 'Unnamed: 0' si existe
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])
    print("\nColumna 'Unnamed: 0' eliminada.")

# Mostrar las primeras filas del DataFrame
print("Primeras filas del DataFrame:")
print(df.head())

# Renombrar columnas para mayor claridad (opcional)
df.columns = [f"Feature_{i}" for i in range(1, len(df.columns) + 1)]
print("\nNuevos nombres de las columnas:")
print(df.columns)

# Resumen estadístico de las variables cuantitativas
print("\nResumen estadístico:")
print(df.describe())

# Normalizar los datos
scaler = StandardScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
print("\nDatos normalizados (primeras filas):")
print(df_normalized.head())

# Crear un mapa de calor para las correlaciones
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Mapa de calor de correlaciones')
plt.show()

# Crear histogramas para las variables
df.hist(figsize=(12, 10), bins=15, color='skyblue', edgecolor='black')
plt.suptitle('Histogramas de las variables')
plt.show()

# Crear un boxplot para detectar valores atípicos
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, palette='Set2')
plt.title('Boxplot de las variables')
plt.xticks(rotation=45)
plt.show()

# Manejo de valores atípicos usando el rango intercuartílico (IQR)
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df_cleaned = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
print(f"\nDimensiones después de eliminar valores atípicos: {df_cleaned.shape}")

# Verificar valores nulos
print("\nValores nulos por columna:")
print(df.isnull().sum())

# Aplicar K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df_cleaned['Cluster'] = kmeans.fit_predict(df_cleaned)
print("\nCentroides de los clusters:")
print(kmeans.cluster_centers_)

# Visualizar los clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_cleaned, x='Feature_1', y='Feature_2', hue='Cluster', palette='viridis')
plt.title('Clusters generados por K-means')
plt.show()

# Dividir los datos en entrenamiento y prueba
X = df_cleaned.drop(columns=['Cluster'])
y = df_cleaned['Cluster']

# Guardar el DataFrame limpio y normalizado en un nuevo archivo CSV
df_cleaned.to_csv('./A07106692_X_cleaned.csv', index=False)
print("\nArchivo limpio guardado como 'A07106692_X_cleaned.csv'.")