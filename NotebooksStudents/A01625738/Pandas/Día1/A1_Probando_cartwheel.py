import pandas as pd

file_path = r'C:\Users\gandh\OneDrive\Escritorio\A01625738\Dia1\cartwheel.csv'
data = pd.read_csv(file_path)

data.index = data.index + 1

filas, columnas = data.shape
print(f"El dataset tiene {filas} filas y {columnas} columnas.\n")

print("Las columnas disponibles son:")
print(list(data.columns))
print()

print(f"Número total de filas en el DataFrame: {len(data)}")

print("\nPrimeras 5 filas del dataset:")
print(data.head())
print("\nÚltimas 5 filas del dataset:")
print(data.tail())

print("\nValores nulos en cada columna:")
print(data.isnull().sum())

print("\nEstadísticas básicas del dataset:")
print(data.describe())

columna = input("\nIngresa el nombre de la columna que deseas ver: ")

if columna in data.columns:
    datos_columna = data[columna]
    print(f"\nDatos de la columna '{columna}':")
    print(datos_columna)
    
    orden = input(f"\n¿Te gustaría ordenar los datos por '{columna}'? (s/n): ").lower()
    if orden == 's':
        asc_desc = input("¿Ascendente (a) o Descendente (d)?: ").lower()
        ascending = True if asc_desc == 'a' else False
        print(f"\nDatos ordenados por '{columna}':")
        print(data.sort_values(by=columna, ascending=ascending))
    
    filtro = input(f"\n¿Te gustaría filtrar los datos por un valor en '{columna}'? (s/n): ").lower()
    if filtro == 's':
        valor = input(f"Ingrese el valor por el cual deseas filtrar en la columna '{columna}': ")
        try:
            valor = float(valor) if datos_columna.dtype != 'object' else valor  # Convertir si no es string
            print(f"\nFiltrando filas donde '{columna}' es igual a {valor}:")
            print(data[data[columna] == valor])
        except ValueError:
            print("\nEl valor ingresado no es válido.")
else:
    print(f"\nLa columna '{columna}' no existe en el dataset.")
