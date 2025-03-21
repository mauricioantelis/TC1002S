# Import the packages that we will be using
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

Ruta            = "C:/Users/rotce/OneDrive/TEC/4 semestre/Semana TEC 1/TC1002S/NotebooksProfessor/cartwheel.csv"
    
df = pd.read_csv(Ruta)

print(df.head())

#tablas de frecuencia
print(df.Glasses.value_counts())

print(df.Glasses.value_counts())
print(df.shape)

#Histograma
plt.hist(df.Age)
plt.show()

plt.hist(df.CWDistance)
plt.show()

plt.hist(df.Age)

plt.hist(df.CWDistance)
plt.show()

g = sns.FacetGrid(df, col="GenderGroup")
g.map(plt.hist, "CWDistance")
plt.show()

#boxplot
plt.boxplot(df.CWDistance)
plt.show()

plt.boxplot(df.CWDistance)
plt.boxplot(df.Wingspan)

plt.show()

# Create side-by-side boxplots of the "CWDistance" grouped by "Gender"
sns.boxplot(data=df, y='CWDistance', hue='Gender', gap=.4)
plt.show()

#scatter plot
plt.scatter(df.Height, df.CWDistance)
plt.show()

plt.scatter(df.Age, df.CWDistance)
plt.scatter(df.Height, df.Wingspan)

plt.show()
