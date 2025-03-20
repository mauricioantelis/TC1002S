# Import the packages that we will be using
import pandas as pd

# Define where you are running the code: colab or local
RunInColab          = True     # (False: no  | True: yes)

# If running in colab:
if RunInColab:
    # Mount your google drive in google colab
    #from google.colab import drive
    #drive.mount('/content/drive')

    # Find location
    #!pwd
    #!ls
    #!ls "/content/drive/My Drive/Colab Notebooks/MachineLearningWithPython/"

    # Define path del proyecto
    Ruta            = "/content/drive/My Drive/Colab Notebooks/MachineLearningWithPython/"

#else:
    # Define path del proyecto
    Ruta            = "C:/Users/rotce/OneDrive/TEC/4 semestre/Semana TEC 1/TC1002S/NotebooksProfessor/cartwheel.csv"
    
df = pd.read_csv(Ruta)

print(type(df))

#rows
print(df.shape[0])
#columns
print(df.shape[1])

#imprime todo
print(df)

#primeras 5 filas
print(df.head())

#ultimas 5 filas
print(df.tail())

#nombre de las columnas
print(df.columns)

#tipo de datos
print(df.dtypes)

#estadisticos para variables cuantitativas
print(df.describe())

print(df.Age.dropna().describe())

#Estadisticas
#promedio
print(df.CWDistance.mean())
#maximo
print(df.CWDistance.max())
#minimo
print(df.CWDistance.min())
#mediana
print(df.CWDistance.median())
#desviacion estandar
print(df.CWDistance.std())

#convertir a excel
df.head().to_csv('primeras5filas.csv')

#Renombrar columnas
df = df.rename(columns={"Age": "Edad"})

print(df.head())

#Seleccionamiento de columans
d = df.iloc[:, 1]

print(d)

print(df[["Gender", "GenderGroup"]])

#.loc
# Return all observations of CWDistance
df.loc[:,"CWDistance"]

# Return a subset of observations of CWDistance
df.loc[:9, "CWDistance"]

# Select all rows for multiple columns, ["Gender", "GenderGroup"]
df.loc[:,["Gender", "GenderGroup"]]

# Select multiple columns, ["Gender", "GenderGroup"]me
keep = ['Gender', 'GenderGroup']
df_gender = df[keep]

# Select few rows for multiple columns, ["CWDistance", "Height", "Wingspan"]
df.loc[4:9, ["CWDistance", "Height", "Wingspan"]]

# Select range of rows for all columns
df.loc[10:15,:]

#iloc
# .
df.iloc[:, :4]

# .
df.iloc[:4, :]

# .
df.iloc[:, 3:7]

# .
df.iloc[4:8, 2:4]

#valores unicoos
print(df.Gender.unique())
print(df.GenderGroup.unique())

#filtros y sorts
print(df[df["Height"] >= 70])
print(df.sort_values("Height"))

print(df.sort_values("Height",ascending=False))

print(df.groupby(['Gender']))#no sale bien 

print(df.groupby(['Gender']).size())

print(df.groupby(['Gender','GenderGroup']).size())

#is null
print(df.isnull().sum())
print(df.notnull().sum())

# Extract all non-missing values of one of the columns into a new variable
x = df.Edad.dropna().describe()
print(x.describe())

#Add/eliminate columns
print(df.head())

# Create a column data
NewColumnData = df.Edad/df.Edad

# Insert that column in the data frame
df.insert(12, "ColumnInserted", NewColumnData, True)

print(df.head())

# # Eliminate inserted column
df.drop(columns=['ColumnInserted'], inplace = True)

print(df.head())
# # Add new column derived from existing columns
#
# # The new column is a function of another column
df["AgeInMonths"] = df["Edad"] * 12
#
print(df.head())
# # Eliminate inserted column
df.drop("AgeInMonths", axis=1, inplace = True)
#
print(df.head())
# Add a new column with text labels reflecting the code's meaning

df["GenderGroupNew"] = df.GenderGroup.replace({1: "Female", 2: "Male"})

print(df.head())
## Eliminate inserted column
df.drop(columns=['GenderGroupNew'],inplace=True)
## Add a new column with strata based on these cut points
#
## Create a column data
NewColumnData = df.Edad/df.Edad
#
## Insert that column in the data frame
df.insert(1, "ColumnStrata", NewColumnData, True)
#
df["ColumnStrata"] = pd.cut(df.Height, [60., 63., 66., 69., 72., 75., 78.])
#
## Show the first 5 rows of the created data frame
print(df.head())
## Eliminate inserted column
df.drop("ColumnStrata", axis=1, inplace = True)
#
print(df.head())
# Drop several "unused" columns
vars = ["ID", "GenderGroup", "GlassesGroup", "CompleteGroup"]
df.drop(vars, axis=1, inplace = True)
print(df.head())

#Agregar Elimiar Filas
print(df.tail())
df.loc[len(df.index)] = [19, 'F', 'Y', 66, 'NaN', 68, 'N', 3]
#
print(df.tail())
## Eliminate inserted row
df.drop([52], inplace = True )
#
print(df.tail())

#limpiando variables
# Drop unused columns
#vars = ["ID", "GenderGroup", "GlassesGroup", "CompleteGroup"]
#df.drop(vars, axis=1, inplace = True)

vars = ["Edad", "Gender", "Glasses", "Height", "Wingspan", "CWDistance", "Complete", "Score"]
df = df[vars]

# Drop rows with any missing values
df = df.dropna()

# Drop unused columns and drop rows with any missing values
vars = ["Edad", "Gender", "Glasses", "Height", "Wingspan", "CWDistance", "Complete", "Score"]
df = df[vars].dropna()

print(df)
