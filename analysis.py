# this programs reads in the data from the iris dataset
# it outputs a summary of each variable
# outputs a scatter plot of each pair of variables
# performs a linear regression to confirm the relationship between any two variables

# i have the iris data in a data subfolder

# author: gerry callaghan

# I will import matplotlib so i can use this library for statisics, plotting etc
import matplotlib.pyplot as plt
# I will import numpy so i can perform various mathematical operations with my arrays 
import numpy as np
#I will import pandas because it is great for with dataframes
import pandas as pd

# first things first is to import the data set
path = "/home/gerry/Downloads/Data_Analytics_Course/pands/pands-project/data/iris/"
#path = "./data/iris/"
csv_filename = path + 'iris.csv'

'''According to the file "iris.names" which was in the zip file, we should get the following
5. Number of Instances: 150 (50 in each of three classes)
6. Number of Attributes: 4 numeric, predictive attributes and the class
7. Attribute Information:
   1. sepal length in cm
   2. sepal width in cm
   3. petal length in cm
   4. petal width in cm
   5. class: 
      -- Iris Setosa
      -- Iris Versicolour
      -- Iris Virginica
'''
# so let's set up some column names in accordance with the Atrribute information
colNames= ('sepal_length',
    'sepal_width', 
    'petal_length', 
    'petal_width', 
    "class", 
)
#create a dataframe (df) and set it equal to our data from the this file iris.csv
# the data is column separated, it currently has no header, let's add the column names as we import the data
df = pd.read_csv(csv_filename, sep=',', header= None, names=colNames)

#'''
#print(f"{df.shape}")
#'''
# the print function has showed us there are 150 rows of data x 5 columns, 

#let's show the data so we can see it has imported okay
#'''
#print(f"{df}") 
#'''
# we can see the first four columns are the sepal/petal length/width and then we have the iris class
# More importantly, we can see that everything has been sorted by the class column, 
# no need for us to sort again

# Let's use drop na to get rid of the values in the series that have no value
    # this actually returns a numpy.ndarray
df.dropna(inplace=True)
#let's show the data again to see if there was any change
#'''
#print(f"{df}") 
#'''
# everything looks fine for us to work with the data now

#now to confirm the number of different classes of iris we use the numpy unique function on the column titled "class"
'''
print(f"{df["class"].nunique()}")
'''
# so there are 3 types, which matches with the attribute information in the data file 
# which says there are Iris Setosa, Iris Versicolour, and Iris Virginica
# we can be more confident that there were no typos in the class names

#Just want to doublecheck the number of instances of class
print(f"{df["class"].shape}")
# there are 150 class observations

'''
# According to the file "iris.names" which was in the zip file, we should get the following
# 5. Number of Instances: 150 (50 in each of three classes)

# Let's confirm
count_setosa = 0
count_versicolor = 0
count_virginica = 0

line_number = 0

while line_number < 150:
    if df["class"][line_number] == "Iris-setosa":
        count_setosa = count_setosa + 1
        line_number = line_number + 1
    elif df["class"][line_number] == "Iris-versicolor":
        count_versicolor = count_versicolor + 1 
        line_number = line_number + 1
    elif df["class"][line_number] == "Iris-virginica":
        count_virginica = count_virginica + 1
        line_number = line_number + 1
    else :
        print(f"You have a typo in line {line_number}")
        line_number = line_number + 1
 
print (f"Iris Class\tObervations \nSetosa:\t\t{count_setosa}\nVersicolor:\t{count_versicolor} \nVirginica:\t{count_virginica}")
'''
# so now we know there are definitely 50 instances of each, which confirms the iris.names.data file

print(f"{df.describe()}")



 



