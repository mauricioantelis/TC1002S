#     # Define path del proyecto
Ruta = "../../NotebooksStudents/A01638902"

# Import the packages that we will be using

# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd
# Dataset url
url = Ruta + "/cartwheel.csv"

# Load the dataset
df = pd.read_csv(url)

# Print the dataset
print(df)