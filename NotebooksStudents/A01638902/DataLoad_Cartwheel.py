#     # Define path del proyecto
Ruta = "../../NotebooksStudents/A01638902"

# Import the packages that we will be using

# import numpy as np
# import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# Dataset url
url = Ruta + "/cartwheel.csv"
url2 = Ruta + "/iris.csv"

# Load the dataset
df = pd.read_csv(url)

# Print the dataset
print(df)

df2 = sns.load_dataset("iris")

print(df2)

#base de datos de load digits

from sklearn import load_digits

DS = load_digits()

print(DS.data.shape)
(1797,64)
import matplotlib.pyplot as plt
plt.gray()
plt.matshow(DS.images[0])
plt.show()

print(DS)