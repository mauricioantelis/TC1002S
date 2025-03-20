import pandas as pd

# Ruta del archivo CSV
ruta = './cartwheel.csv'
df = pd.read_csv(ruta)

# Mostrar las primeras filas del DataFrame
print("Primeras filas del DataFrame:")
print(df.head())

# Mostrar las ultimas filas del DataFrame
print("Ultimas filas del DataFrame:")
print(df.tail())

# Resumen estadístico de las variables cuantitativas
print("\nResumen estadístico:")
print(df.describe())

# Mostrar el número de filas y columnas
print("\nDimensiones del DataFrame (filas, columnas):")
print(df.shape)

# Mostrar los nombres de las columnas
print("\nNombres de las columnas:")
print(df.columns)

# Mostrar los tipos de datos de cada columna
print("\nTipos de datos de las columnas:")
print(df.dtypes)

# Verificar valores únicos en una columna (Gender)
if 'Gender' in df.columns:
    print("\nValores únicos en la columna 'Gender':")
    print(df['Gender'].unique())

# Verificar valores nulos en el DataFrame
print("\nValores nulos por columna:")
print(df.isnull().sum())

# Crear un nuevo DataFrame sin valores nulos
df_clean = df.dropna()
print("\nDataFrame limpio (sin valores nulos):")
print(df_clean.head())

# Crear un nuevo DataFrame con columnas específicas (Height y CWDistance)
if 'Height' in df.columns and 'CWDistance' in df.columns:
    df_subset = df[['Height', 'CWDistance']]
    print("\nSubset del DataFrame con columnas 'Height' y 'CWDistance':")
    print(df_subset.head())

# Agregar una nueva columna calculada (Altura en centímetros si está en pulgadas)
if 'Height' in df.columns:
    df['Height_cm'] = df['Height'] * 2.54
    print("\nDataFrame con nueva columna 'Height_cm':")
    print(df[['Height', 'Height_cm']].head())

# Guardar el DataFrame limpio en un nuevo archivo CSV
df_clean.to_csv('./cartwheel_clean.csv', index=False)
print("\nArchivo limpio guardado como 'cartwheel_clean.csv'.")
