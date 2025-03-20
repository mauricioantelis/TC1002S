import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar el archivo CSV
ruta = 'cartwheel.csv'
df = pd.read_csv(ruta)

# Número de veces que cada valor distinto ocurre en un conjunto de datos
print(df.value_counts())

# Proporción de cada valor distinto en un conjunto de datos
print(df.value_counts(normalize=True))

# Total de observaciones
print(f"Total de observaciones: {len(df)}")

# Histograma de una variable (Age)
plt.figure(figsize=(8, 6))
sns.histplot(df['Age'], kde=True, bins=10, color='blue')
plt.title('Histograma de Age')
plt.xlabel('Age')
plt.ylabel('Frecuencia')
plt.show()

# Boxplot de una variable (CWDistance)
plt.figure(figsize=(8, 6))
sns.boxplot(y=df['CWDistance'], color='green')
plt.title('Boxplot de CWDistance')
plt.ylabel('CWDistance')
plt.show()

# Scatter plot entre dos variables (Height' y 'Wingspan')
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['Height'], y=df['Wingspan'], hue=df['Gender'])
plt.title('Scatter plot entre Height y Wingspan')
plt.xlabel('Height')
plt.ylabel('Wingspan')
plt.legend(title='Gender')
plt.show()

# Pairplot de múltiples variables
sns.pairplot(df[['Age', 'CWDistance', 'Height', 'Wingspan']], diag_kind='kde')
plt.show()