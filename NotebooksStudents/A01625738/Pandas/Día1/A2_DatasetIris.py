import pandas as pd

# Cargar el archivo CSV
file_path = r'C:\Users\gandh\OneDrive\Escritorio\A01625738\Día1\iris.csv'
data = pd.read_csv(file_path)

# Punto 1: Calcular el resumen estadístico para cada variable cuantitativa
print("Resumen estadístico del dataset:")
print(data.describe())

# Identificar los nombres de las columnas y tipos de datos
print("\nNombres de las columnas:")
print(data.columns)

print("\nTipos de datos de cada columna:")
print(data.dtypes)

# Mínimos, máximos, media, mediana y desviación estándar
print("\nValores mínimos:")
print(data.min())

print("\nValores máximos:")
print(data.max())

# Calcular la media solo para las columnas numéricas
print("\nMedia de las columnas numéricas:")
print(data.select_dtypes(include='number').mean())

# Calcular la mediana solo para las columnas numéricas
print("\nMediana de las columnas numéricas:")
print(data.select_dtypes(include='number').median())

# Calcular la desviación estándar solo para las columnas numéricas
print("\nDesviación estándar de las columnas numéricas:")
print(data.select_dtypes(include='number').std())

# Punto 2: Verificar si hay datos faltantes y crear un nuevo dataset sin valores nulos
print("\nValores nulos en cada columna:")
print(data.isnull().sum())

# Crear un nuevo dataset sin datos faltantes
data_cleaned = data.dropna()
print(f"\nNúmero de filas en el dataset limpio (sin valores nulos): {len(data_cleaned)}")

# Punto 3: Crear un nuevo dataset con solo el ancho y largo de los pétalos y el tipo de flor
petal_data = data[['petal.length', 'petal.width', 'variety']]
print("\nDataset con el ancho y largo de los pétalos y el tipo de flor:")
print(petal_data.head())

# Punto 4: Crear un nuevo dataset con solo el ancho y largo de los sépalos y el tipo de flor
sepal_data = data[['sepal.length', 'sepal.width', 'variety']]
print("\nDataset con el ancho y largo de los sépalos y el tipo de flor:")
print(sepal_data.head())

# Punto 5: Crear un nuevo dataset con los sépalos y la flor codificada numéricamente
data['variety_encoded'] = data['variety'].astype('category').cat.codes
sepal_encoded_data = data[['sepal.length', 'sepal.width', 'variety_encoded']]
print("\nDataset con el ancho y largo de los sépalos y el tipo de flor codificado:")
print(sepal_encoded_data.head())
