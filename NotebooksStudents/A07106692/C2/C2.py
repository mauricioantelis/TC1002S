# Importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Importing data
colnames = ["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width", "Flower"]
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pd.read_csv(url, header=None, names=colnames)

print(df)