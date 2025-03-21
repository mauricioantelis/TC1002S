import pandas as pd

Ruta = "../../NotebooksStudents/A01638902"

url = Ruta + "/cartwheel.csv"
urlIris = Ruta + "/iris.csv"

df = pd.read_csv(url)

print(type(df))

numRows = df.shape[0]
numColumns = df.shape[1]

rows_cols = df.shape

print(numRows)
print(numColumns)


print(rows_cols)


print(df.head()) #Para Mostrar unicamente los primeros 5 datos del df

print(df.tail()) #Imprime los ultimos 5

print(df.columns) #Muestra el nombre de las columnas en el dataset

print(df.dtypes) #Muestra los tipos de datos (float, int etc) del df

# Summary statistics for the quantitative variables

# Drop observations with NaN values

AgeColumn = df.Age.dropna()
WingspanColumn = df.Wingspan.dropna()

print(AgeColumn.describe()) #dropNA
print(WingspanColumn.describe())


print(AgeColumn.mean())
print(df.corr(numeric_only=True))
print(AgeColumn.max())
print(AgeColumn.min())
print(AgeColumn.median())
print(AgeColumn.std())

df.to_csv("myDataFrame.csv")
df.to_csv("myDataFrame.csv", sep="\t")


#Rename Columns

df=df.rename(columns={"Age":"Edad"})
print(df.head())

df = df.rename(columns={"Edad" : "Age"})
print(df.head())


a = df.Age
b = df["Age"]
c = df.loc[:, "Age"]
d = df.iloc[:, 1]
print(d)
df[["Gender", "GenderGroup"]]


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

print(df.Gender.unique())
print(df["GenderGroup"])

print(df[df["Height"] >= 70].head())
print(df.sort_values("Height"))
print(df.sort_values("Height", ascending=False))

print(df.groupby(["Gender"]))
print(df.groupby(["Gender"]).size())

print(df.isnull().sum())
print(df.notnull().sum())
print( df.Height.notnull().sum() )
print( pd.isnull(df.Height).sum() )


# Extract all non-missing values of one of the columns into a new variable
x = df.Age.dropna().describe()
x.describe()

#Add and eleminate columns

df.head()


# Add a new column with new data
NewColumnData = 5
# Create a column data
#NewColumnData = df.Age/df.Age

# Insert that column in the data frame
df.insert(12, "ColumnInserted", NewColumnData, True)

print(df.head())

# # Eliminate inserted column
# # # Remove three columns as index base
df.drop(df.columns[[12]], axis = 1, inplace = True)
# #
df.head()



# # Add new column derived from existing columns
#
# # The new column is a function of another column
df["AgeInMonths"] = df["Age"] * 12
#
#
#
df.head()



# # Eliminate inserted column
df.drop("AgeInMonths", axis=1, inplace = True)
#
df.head()

# Add a new column with text labels reflecting the code's meaning

df["GenderGroupNew"] = df.GenderGroup.replace({1: "Female", 2: "Male"})

# Show the first 5 rows of the created data frame

df.head()

## Eliminate inserted column
df.drop("GenderGroupNew", axis=1, inplace = True)
##df.drop(['GenderGroupNew'],vaxis='columns',vinplace=True)


## Add a new column with strata based on these cut points
#
## Create a column data
NewColumnData = df.Age/df.Age
#
## Insert that column in the data frame
df.insert(1, "ColumnStrata", NewColumnData, True)
#
df["ColumnStrata"] = pd.cut(df.Height, [60., 63., 66., 69., 72., 75., 78.])
#
## Show the first 5 rows of the created data frame
df.head()
print(df)


## Eliminate inserted column

df.drop("ColumnStrata", axis=1, inplace = True)
#
df.head()

print(df)

# Drop several "unused" columns
vars = ["ID", "GenderGroup", "GlassesGroup", "CompleteGroup"]
df.drop(vars, axis=1, inplace = True)



# df.loc[len(df.index)] = [26, 24, 'F', 1, 'Y', 1, 66, 'NaN', 68, 'N', 0, 3]
# #
# df.tail()


## Eliminate inserted row
df.drop([28], inplace = True )
#
df.tail()

# Drop unused columns
# vars = ["ID", "GenderGroup", "GlassesGroup", "CompleteGroup"]
# df.drop(vars, axis=1, inplace = True)

vars = ["Age", "Gender", "Glasses", "Height", "Wingspan", "CWDistance", "Complete", "Score"]
df = df[vars]

# # Drop rows with any missing values
# df = df.dropna()

# # Drop unused columns and drop rows with any missing values
# vars = ["Age", "Gender", "Glasses", "Height", "Wingspan", "CWDistance", "Complete", "Score"]
# df = df[vars].dropna()

print(df)


#ACtividad 


import pandas as pd

# Cargar el nuevo dataset
df = pd.read_csv(urlIris)

# Mostrar nombres y tipos de cada columna
print(df.dtypes)

summary = df.describe()
print("\nResumen estadístico de las variables cuantitativas:")
print(summary)

missing_values = df.isnull().sum()
print("Valores faltantes")
print(missing_values)

df_clean = df.dropna()
df_clean.to_csv("iris_clean.csv", index=False)
print("Dataset limpio")

df_petal = df_clean[['petal.width', 'petal.length', 'variety']]
df_petal.to_csv("iris_petal.csv", index=False)
print("Dataset con ancho y largo de pétalos y tipo de flor")

df_sepal = df_clean[['sepal.width', 'sepal.length', 'variety']]
df_sepal.to_csv("iris_sepal.csv", index=False)
print("Dataset con ancho y largo de sépalos y tipo de flor")

df_sepal_encoded = df_sepal.copy()
df_sepal_encoded['variety'] = df_sepal_encoded['variety'].astype('category').cat.codes
df_sepal_encoded.to_csv("iris_sepal_encoded.csv", index=False)
print("Dataset con ancho y largo de sépalos y tipo de flor aplicado numericamente")
