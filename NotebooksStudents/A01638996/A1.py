# Define where you are running the code: colab or local
RunInColab          = False     # (False: no  | True: yes)

# If running in colab:
if RunInColab:
    # Mount your google drive in google colab
    from google.colab import drive
    drive.mount('/content/drive')

    # Find location
    #!pwd
    #!ls
    #!ls "/content/drive/My Drive/Colab Notebooks/MachineLearningWithPython/"

    # Define path del proyecto
    Ruta            = "/content/drive/My Drive/Colab Notebooks/MachineLearningWithPython/"

#else:
    # Define path del proyecto
   # Ruta = "C:\Users\rotce\OneDrive\TEC\4 semestre\Semana TEC 1\TC1002S\NotebooksStudents\A01638996"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

url = "C:/Users/rotce/OneDrive/TEC/4 semestre/Semana TEC 1/TC1002S/NotebooksProfessor/cartwheel.csv"

#base de datos Cartwheel
df = pd.read_csv(url)

print(df)

#impresion de columnas
df.shape[1]
#impresion de filas
df.shape[2]

#base de datos IRIS
df2 = sns.load_dataset("iris")

print("ESPACIO")

print(df2)

#base de datos load digits
from sklearn.datasets import load_digits

DS = load_digits()

print(DS.data.shape)
(1797, 64)
import matplotlib.pyplot as plt
plt.gray()
plt.matshow(DS.images[0])
plt.show()

print(DS)
