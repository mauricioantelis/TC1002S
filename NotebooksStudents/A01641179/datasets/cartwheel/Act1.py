# Define path del proyecto
#Ruta            = "C:\Users\artur\Downloads\TC1002S\NotebooksStudents\A01641179\datasets\cartwheel"

import csv
import pandas as pd

with open('cartwheel.csv') as csvfile:
    
    #row_count = sum(1 for row in csvfile)
    csvReader = csv.reader(csvfile, delimiter=',')
    for row in csvReader:
        print(row)
        #print(row_count-1)

df = pd.read_csv("cartwheel.csv") #===> reads in all the rows, but skips the first one as it is a header..

total_rows=len(df.axes[0]) #===> Axes of 0 is for a row
total_cols=len(df.axes[1]) #===> Axes of 0 is for a column
print("Numero de filas: "+str(total_rows))
print("Numero de columnas: "+str(total_cols))

