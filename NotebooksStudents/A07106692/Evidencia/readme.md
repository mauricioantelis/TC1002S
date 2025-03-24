# Evidencia - Análisis y Procesamiento de Datos

Este proyecto realiza un análisis y procesamiento de datos utilizando Python. A continuación, se describe el flujo de trabajo implementado en el archivo `main.py`:

## Descripción del Script

El archivo `main.py` contiene un pipeline de análisis de datos que incluye las siguientes etapas:

1. **Carga de Datos**:
   - Se carga un archivo CSV (`A07106692_X.csv`) en un DataFrame de pandas.
   - Se elimina la columna `Unnamed: 0` si está presente.

2. **Renombrado de Columnas**:
   - Las columnas del DataFrame se renombran para mayor claridad, utilizando el formato `Feature_1`, `Feature_2`, etc.

3. **Exploración de Datos**:
   - Se genera un resumen estadístico de las variables cuantitativas.
   - Se verifica la existencia de valores nulos.

4. **Visualización de Datos**:
   - **Mapa de Calor**:
     - Se genera un mapa de calor para visualizar las correlaciones entre las variables del DataFrame.
     - Este gráfico utiliza colores para representar la intensidad de las correlaciones, donde valores cercanos a 1 o -1 indican una fuerte relación positiva o negativa, respectivamente.
     - Es útil para identificar patrones y relaciones entre las variables.
   - **Histogramas**:
     - Se crean histogramas para cada variable del DataFrame.
     - Los histogramas muestran la distribución de los datos, permitiendo identificar si los valores están concentrados en un rango específico o si hay sesgos.
     - Ayudan a detectar posibles problemas como distribuciones no normales.
   - **Boxplots**:
     - Se generan diagramas de caja (boxplots) para cada variable.
     - Estos gráficos muestran la mediana, los cuartiles y los valores atípicos de las variables.
     - Son útiles para identificar valores extremos que podrían afectar el análisis.

5. **Manejo de Valores Atípicos**:
   - Se eliminan valores atípicos utilizando el rango intercuartílico (IQR).

6. **Normalización de Datos**:
   - Los datos se normalizan utilizando `StandardScaler` de scikit-learn.
   - **¿Qué significa normalizar?**: Este proceso ajusta los valores de las características para que tengan una media de 0 y una desviación estándar de 1. Esto asegura que todas las variables estén en la misma escala.
   - **¿Por qué es importante?**: En la práctica, la normalización mejora el rendimiento de algoritmos sensibles a las magnitudes de las variables, como K-means, evitando que las características con valores más grandes dominen el análisis.

7. **Clustering con K-means**:
   - Se aplica el algoritmo K-means para agrupar los datos en 3 clusters.
   - **Visualización de Clusters**:
     - Se genera un gráfico de dispersión para visualizar los clusters generados.
     - Los puntos se colorean según el cluster al que pertenecen, permitiendo observar cómo se agrupan los datos en el espacio de las primeras dos características (`Feature_1` y `Feature_2`).

8. **Exportación de Datos**:
   - El DataFrame limpio y normalizado se guarda en un nuevo archivo CSV (`A07106692_X_cleaned.csv`).

## Requisitos

Para ejecutar este script, es necesario tener instaladas las siguientes bibliotecas de Python:
- pandas
- seaborn
- matplotlib
- scikit-learn

## Ejecución

1. Coloca el archivo `A07106692_X.csv` en el mismo directorio que el script.
2. Ejecuta el script `main.py` en tu entorno de Python.
3. Revisa las visualizaciones generadas y el archivo limpio exportado (`A07106692_X_cleaned.csv`).

## Resultados

El script proporciona:
- **Mapa de calor** para identificar correlaciones entre variables.
- **Histogramas** para analizar la distribución de los datos.
- **Boxplots** para detectar valores atípicos.
- **Gráfico de dispersión** para visualizar los clusters generados por K-means.
- Un conjunto de datos limpio y normalizado.
- Clusters generados por K-means.

Este pipeline es útil para realizar análisis exploratorio y preparar los datos para tareas de machine learning.

