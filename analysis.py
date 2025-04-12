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
colNames= ("sepal_length",
    "sepal_width", 
    "petal_length", 
    "petal_width", 
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

#summary=df.describe()
#print(f"{summary}")

#A function for creating arrays for each variable
def creating_array(DataFrame,min,max):
    row_number = min
    output_array = []
    
    while row_number < max:
         current_array = df["sepal_length"][row_number],df["sepal_width"][row_number],df["petal_length"][row_number],df["petal_width"][row_number]
         row_number = row_number + 1
         output_array = output_array + [current_array] 
    #current_array = df["sepal_length"][row_number],df["sepal_width"][row_number],df["petal_length"][row_number],df["petal_width"][row_number]
    #output_array = output_array + [current_array] 
    return output_array

# so now I'm going to try creating an array for each class, whereby I will call the function to incrementally append each line
# to the existing array, the append function won't work for me , so I'm going to do what i would do in Visual Basic and do it long hand 
count_setosa = 50
count_versicolor = 50
count_virginica = 50
setosa = []
versicolor = []
virginica =[]
  
setosa = creating_array(df, min=0,max=count_setosa)
versicolor = creating_array(df, min=50,max=100)
virginica = creating_array(df, min=100,max=150)

#print(f"The array for Setosa is:\n{setosa}\n\nThe array for Versicolor is:\n{versicolor}\n\nThe array for Virginica is:\n{virginica}")
# We want to output a summary of each variable 
iris = [setosa,versicolor,virginica]
#print(f"{iris}")
setosa_sepal_length=(iris[0][0][0],iris[0][1][0],iris[0][2][0],iris[0][3][0],iris[0][4][0],iris[0][5][0],iris[0][6][0],iris[0][7][0],iris[0][8][0],iris[0][9][0],iris[0][10][0],iris[0][11][0],iris[0][12][0],iris[0][13][0],iris[0][14][0],iris[0][15][0],iris[0][16][0],iris[0][17][0],iris[0][18][0],iris[0][19][0],iris[0][20][0],iris[0][21][0],iris[0][22][0],iris[0][23][0],iris[0][24][0],iris[0][25][0],iris[0][26][0],iris[0][27][0],iris[0][28][0],iris[0][29][0],iris[0][30][0],iris[0][31][0],iris[0][32][0],iris[0][33][0],iris[0][34][0],iris[0][35][0],iris[0][36][0],iris[0][37][0],iris[0][38][0],iris[0][39][0],iris[0][40][0],iris[0][41][0],iris[0][42][0],iris[0][43][0],iris[0][44][0],iris[0][45][0],iris[0][46][0],iris[0][47][0],iris[0][48][0],iris[0][49][0])
#print(f"Setosa Sepal Lengths are: {setosa_sepal_length}")
versicolor_sepal_length=(iris[1][0][0],iris[1][1][0],iris[1][2][0],iris[1][3][0],iris[1][4][0],iris[1][5][0],iris[1][6][0],iris[1][7][0],iris[1][8][0],iris[1][9][0],iris[1][10][0],iris[1][11][0],iris[1][12][0],iris[1][13][0],iris[1][14][0],iris[1][15][0],iris[1][16][0],iris[1][17][0],iris[1][18][0],iris[1][19][0],iris[1][20][0],iris[1][21][0],iris[1][22][0],iris[1][23][0],iris[1][24][0],iris[1][25][0],iris[1][26][0],iris[1][27][0],iris[1][28][0],iris[1][29][0],iris[1][30][0],iris[1][31][0],iris[1][32][0],iris[1][33][0],iris[1][34][0],iris[1][35][0],iris[1][36][0],iris[1][37][0],iris[1][38][0],iris[1][39][0],iris[1][40][0],iris[1][41][0],iris[1][42][0],iris[1][43][0],iris[1][44][0],iris[1][45][0],iris[1][46][0],iris[1][47][0],iris[1][48][0],iris[1][49][0])
#print(f"Versicolor Sepal Lengths are: {versicolor_sepal_length}")
virginica_sepal_length=(iris[2][0][0],iris[2][1][0],iris[2][2][0],iris[2][3][0],iris[2][4][0],iris[2][5][0],iris[2][6][0],iris[2][7][0],iris[2][8][0],iris[2][9][0],iris[2][10][0],iris[2][11][0],iris[2][12][0],iris[2][13][0],iris[2][14][0],iris[2][15][0],iris[2][16][0],iris[2][17][0],iris[2][18][0],iris[2][19][0],iris[2][20][0],iris[2][21][0],iris[2][22][0],iris[2][23][0],iris[2][24][0],iris[2][25][0],iris[2][26][0],iris[2][27][0],iris[2][28][0],iris[2][29][0],iris[2][30][0],iris[2][31][0],iris[2][32][0],iris[2][33][0],iris[2][34][0],iris[2][35][0],iris[2][36][0],iris[2][37][0],iris[2][38][0],iris[2][39][0],iris[2][40][0],iris[2][41][0],iris[2][42][0],iris[2][43][0],iris[2][44][0],iris[2][45][0],iris[2][46][0],iris[2][47][0],iris[2][48][0],iris[2][49][0])
#print(f"Virginica Sepal Lengths are: {virginica_sepal_length}")


fig, ax = plt.subplots()
ax.hist(setosa_sepal_length, bins=10, edgecolor="black", label="Setosa", alpha=0.5)
ax.hist(versicolor_sepal_length, bins=10, edgecolor="black", label="Versicolor", alpha=0.5)
ax.hist(virginica_sepal_length, bins=10, edgecolor="black", label="Virginica", alpha=0.5)
ax.legend()
ax.set_xlabel("Length of Sepal (cm)")
ax.set_ylabel("Frequency")
plt.title("Sepal Length")
#plt.show()


# Descriptive Statistics
print(f"\nSepal Lengths")
print(f"\t\tSetosa\t\tVersicolor\tVirginica")
print(f"Mean:\t\t{round(np.mean(setosa_sepal_length),2)}\t\t{round(np.mean(versicolor_sepal_length),2)}\t\t{round(np.mean(virginica_sepal_length),2)}")
print(f"St.Dev:\t\t{round(np.std(setosa_sepal_length),2)}\t\t{round(np.std(versicolor_sepal_length),2)}\t\t{round(np.std(virginica_sepal_length),2)}")
print(f"Median:\t\t{round(np.median(setosa_sepal_length),2)}\t\t{round(np.median(versicolor_sepal_length),2)}\t\t{round(np.median(virginica_sepal_length),2)}")
print(f"Maximum:\t{round(np.max(setosa_sepal_length),2)}\t\t{round(np.max(versicolor_sepal_length),2)}\t\t{round(np.max(virginica_sepal_length),2)}")
print(f"Minimum:\t{round(np.min(setosa_sepal_length),2)}\t\t{round(np.min(versicolor_sepal_length),2)}\t\t{round(np.min(virginica_sepal_length),2)}")


setosa_sepal_width=(iris[0][0][1],iris[0][1][1],iris[0][2][1],iris[0][3][1],iris[0][4][1],iris[0][5][1],iris[0][6][1],iris[0][7][1],iris[0][8][1],iris[0][9][1],iris[0][10][1],iris[0][11][1],iris[0][12][1],iris[0][13][1],iris[0][14][1],iris[0][15][1],iris[0][16][1],iris[0][17][1],iris[0][18][1],iris[0][19][1],iris[0][20][1],iris[0][21][1],iris[0][22][1],iris[0][23][1],iris[0][24][1],iris[0][25][1],iris[0][26][1],iris[0][27][1],iris[0][28][1],iris[0][29][1],iris[0][30][1],iris[0][31][1],iris[0][32][1],iris[0][33][1],iris[0][34][1],iris[0][35][1],iris[0][36][1],iris[0][37][1],iris[0][38][1],iris[0][39][1],iris[0][40][1],iris[0][41][1],iris[0][42][1],iris[0][43][1],iris[0][44][1],iris[0][45][1],iris[0][46][1],iris[0][47][1],iris[0][48][1],iris[0][49][1])
#print(f"Setosa Sepal Widths are: {setosa_sepal_width}")
versicolor_sepal_width=(iris[1][0][1],iris[1][1][1],iris[1][2][1],iris[1][3][1],iris[1][4][1],iris[1][5][1],iris[1][6][1],iris[1][7][1],iris[1][8][1],iris[1][9][1],iris[1][10][1],iris[1][11][1],iris[1][12][1],iris[1][13][1],iris[1][14][1],iris[1][15][1],iris[1][16][1],iris[1][17][1],iris[1][18][1],iris[1][19][1],iris[1][20][1],iris[1][21][1],iris[1][22][1],iris[1][23][1],iris[1][24][1],iris[1][25][1],iris[1][26][1],iris[1][27][1],iris[1][28][1],iris[1][29][1],iris[1][30][1],iris[1][31][1],iris[1][32][1],iris[1][33][1],iris[1][34][1],iris[1][35][1],iris[1][36][1],iris[1][37][1],iris[1][38][1],iris[1][39][1],iris[1][40][1],iris[1][41][1],iris[1][42][1],iris[1][43][1],iris[1][44][1],iris[1][45][1],iris[1][46][1],iris[1][47][1],iris[1][48][1],iris[1][49][1])
#print(f"Versicolor Sepal Widths are: {versicolor_sepal_width}")
virginica_sepal_width=(iris[2][0][1],iris[2][1][1],iris[2][2][1],iris[2][3][1],iris[2][4][1],iris[2][5][1],iris[2][6][1],iris[2][7][1],iris[2][8][1],iris[2][9][1],iris[2][10][1],iris[2][11][1],iris[2][12][1],iris[2][13][1],iris[2][14][1],iris[2][15][1],iris[2][16][1],iris[2][17][1],iris[2][18][1],iris[2][19][1],iris[2][20][1],iris[2][21][1],iris[2][22][1],iris[2][23][1],iris[2][24][1],iris[2][25][1],iris[2][26][1],iris[2][27][1],iris[2][28][1],iris[2][29][1],iris[2][30][1],iris[2][31][1],iris[2][32][1],iris[2][33][1],iris[2][34][1],iris[2][35][1],iris[2][36][1],iris[2][37][1],iris[2][38][1],iris[2][39][1],iris[2][40][1],iris[2][41][1],iris[2][42][1],iris[2][43][1],iris[2][44][1],iris[2][45][1],iris[2][46][1],iris[2][47][1],iris[2][48][1],iris[2][49][1])
#print(f"Virginica Sepal Widths are: {virginica_sepal_width}")

fig, ax = plt.subplots()
ax.hist(setosa_sepal_width, bins=10, edgecolor="black", label="Setosa", alpha=0.5)
ax.hist(versicolor_sepal_width, bins=10, edgecolor="black", label="Versicolor", alpha=0.5)
ax.hist(virginica_sepal_width, bins=10, edgecolor="black", label="Virginica", alpha=0.5)
ax.legend()
ax.set_xlabel("Width of Sepal (cm)")
ax.set_ylabel("Frequency")
plt.title("Sepal Width")
#plt.show()

# Descriptive Statistics
print(f"\nSepal Width")
print(f"\t\tSetosa\t\tVersicolor\tVirginica")
print(f"Mean:\t\t{round(np.mean(setosa_sepal_width),2)}\t\t{round(np.mean(versicolor_sepal_width),2)}\t\t{round(np.mean(virginica_sepal_width),2)}")
print(f"St.Dev:\t\t{round(np.std(setosa_sepal_width),2)}\t\t{round(np.std(versicolor_sepal_width),2)}\t\t{round(np.std(virginica_sepal_width),2)}")
print(f"Median:\t\t{round(np.median(setosa_sepal_width),2)}\t\t{round(np.median(versicolor_sepal_width),2)}\t\t{round(np.median(virginica_sepal_width),2)}")
print(f"Maximum:\t{round(np.max(setosa_sepal_width),2)}\t\t{round(np.max(versicolor_sepal_width),2)}\t\t{round(np.max(virginica_sepal_width),2)}")
print(f"Minimum:\t{round(np.min(setosa_sepal_width),2)}\t\t{round(np.min(versicolor_sepal_width),2)}\t\t{round(np.min(virginica_sepal_width),2)}")

setosa_petal_length=(iris[0][0][2],iris[0][1][2],iris[0][2][2],iris[0][3][2],iris[0][4][2],iris[0][5][2],iris[0][6][2],iris[0][7][2],iris[0][8][2],iris[0][9][2],iris[0][10][2],iris[0][11][2],iris[0][12][2],iris[0][13][2],iris[0][14][2],iris[0][15][2],iris[0][16][2],iris[0][17][2],iris[0][18][2],iris[0][19][2],iris[0][20][2],iris[0][21][2],iris[0][22][2],iris[0][23][2],iris[0][24][2],iris[0][25][2],iris[0][26][2],iris[0][27][2],iris[0][28][2],iris[0][29][2],iris[0][30][2],iris[0][31][2],iris[0][32][2],iris[0][33][2],iris[0][34][2],iris[0][35][2],iris[0][36][2],iris[0][37][2],iris[0][38][2],iris[0][39][2],iris[0][40][2],iris[0][41][2],iris[0][42][2],iris[0][43][2],iris[0][44][2],iris[0][45][2],iris[0][46][2],iris[0][47][2],iris[0][48][2],iris[0][49][2])
#print(f"Setosa Petal Lengths are: {setosa_petal_length}")
versicolor_petal_length=(iris[1][0][2],iris[1][1][2],iris[1][2][2],iris[1][3][2],iris[1][4][2],iris[1][5][2],iris[1][6][2],iris[1][7][2],iris[1][8][2],iris[1][9][2],iris[1][10][2],iris[1][11][2],iris[1][12][2],iris[1][13][2],iris[1][14][2],iris[1][15][2],iris[1][16][2],iris[1][17][2],iris[1][18][2],iris[1][19][2],iris[1][20][2],iris[1][21][2],iris[1][22][2],iris[1][23][2],iris[1][24][2],iris[1][25][2],iris[1][26][2],iris[1][27][2],iris[1][28][2],iris[1][29][2],iris[1][30][2],iris[1][31][2],iris[1][32][2],iris[1][33][2],iris[1][34][2],iris[1][35][2],iris[1][36][2],iris[1][37][2],iris[1][38][2],iris[1][39][2],iris[1][40][2],iris[1][41][2],iris[1][42][2],iris[1][43][2],iris[1][44][2],iris[1][45][2],iris[1][46][2],iris[1][47][2],iris[1][48][2],iris[1][49][2])
#print(f"Versicolor Petal Lengths are: {versicolor_petal_length}")
virginica_petal_length=(iris[2][0][2],iris[2][1][2],iris[2][2][2],iris[2][3][2],iris[2][4][2],iris[2][5][2],iris[2][6][2],iris[2][7][2],iris[2][8][2],iris[2][9][2],iris[2][10][2],iris[2][11][2],iris[2][12][2],iris[2][13][2],iris[2][14][2],iris[2][15][2],iris[2][16][2],iris[2][17][2],iris[2][18][2],iris[2][19][2],iris[2][20][2],iris[2][21][2],iris[2][22][2],iris[2][23][2],iris[2][24][2],iris[2][25][2],iris[2][26][2],iris[2][27][2],iris[2][28][2],iris[2][29][2],iris[2][30][2],iris[2][31][2],iris[2][32][2],iris[2][33][2],iris[2][34][2],iris[2][35][2],iris[2][36][2],iris[2][37][2],iris[2][38][2],iris[2][39][2],iris[2][40][2],iris[2][41][2],iris[2][42][2],iris[2][43][2],iris[2][44][2],iris[2][45][2],iris[2][46][2],iris[2][47][2],iris[2][48][2],iris[2][49][2])
#print(f"Virginica Petal Lengths are: {virginica_petal_length}")


fig, ax = plt.subplots()
ax.hist(setosa_petal_length, bins=10, edgecolor="black", label="Setosa", alpha=0.5)
ax.hist(versicolor_petal_length, bins=10, edgecolor="black", label="Versicolor", alpha=0.5)
ax.hist(virginica_petal_length, bins=10, edgecolor="black", label="Virginica", alpha=0.5)
ax.legend()
ax.set_xlabel("Length of Petal (cm)")
ax.set_ylabel("Frequency")
plt.title("Petal Length")
#plt.show()

# Descriptive Statistics
print(f"\nPetal Length")
print(f"\t\tSetosa\t\tVersicolor\tVirginica")
print(f"Mean:\t\t{round(np.mean(setosa_petal_length),2)}\t\t{round(np.mean(versicolor_petal_length),2)}\t\t{round(np.mean(virginica_petal_length),2)}")
print(f"St.Dev:\t\t{round(np.std(setosa_petal_length),2)}\t\t{round(np.std(versicolor_petal_length),2)}\t\t{round(np.std(virginica_petal_length),2)}")
print(f"Median:\t\t{round(np.median(setosa_petal_length),2)}\t\t{round(np.median(versicolor_petal_length),2)}\t\t{round(np.median(virginica_petal_length),2)}")
print(f"Maximum:\t{round(np.max(setosa_petal_length),2)}\t\t{round(np.max(versicolor_petal_length),2)}\t\t{round(np.max(virginica_petal_length),2)}")
print(f"Minimum:\t{round(np.min(setosa_petal_length),2)}\t\t{round(np.min(versicolor_petal_length),2)}\t\t{round(np.min(virginica_petal_length),2)}")

setosa_petal_width=(iris[0][0][3],iris[0][1][3],iris[0][2][3],iris[0][3][3],iris[0][4][3],iris[0][5][3],iris[0][6][3],iris[0][7][3],iris[0][8][3],iris[0][9][3],iris[0][10][3],iris[0][11][3],iris[0][12][3],iris[0][13][3],iris[0][14][3],iris[0][15][3],iris[0][16][3],iris[0][17][3],iris[0][18][3],iris[0][19][3],iris[0][20][3],iris[0][21][3],iris[0][22][3],iris[0][23][3],iris[0][24][3],iris[0][25][3],iris[0][26][3],iris[0][27][3],iris[0][28][3],iris[0][29][3],iris[0][30][3],iris[0][31][3],iris[0][32][3],iris[0][33][3],iris[0][34][3],iris[0][35][3],iris[0][36][3],iris[0][37][3],iris[0][38][3],iris[0][39][3],iris[0][40][3],iris[0][41][3],iris[0][42][3],iris[0][43][3],iris[0][44][3],iris[0][45][3],iris[0][46][3],iris[0][47][3],iris[0][48][3],iris[0][49][3])
#print(f"Setosa Petal Widths are: {setosa_petal_width}")
versicolor_petal_width=(iris[1][0][3],iris[1][1][3],iris[1][2][3],iris[1][3][3],iris[1][4][3],iris[1][5][3],iris[1][6][3],iris[1][7][3],iris[1][8][3],iris[1][9][3],iris[1][10][3],iris[1][11][3],iris[1][12][3],iris[1][13][3],iris[1][14][3],iris[1][15][3],iris[1][16][3],iris[1][17][3],iris[1][18][3],iris[1][19][3],iris[1][20][3],iris[1][21][3],iris[1][22][3],iris[1][23][3],iris[1][24][3],iris[1][25][3],iris[1][26][3],iris[1][27][3],iris[1][28][3],iris[1][29][3],iris[1][30][3],iris[1][31][3],iris[1][32][3],iris[1][33][3],iris[1][34][3],iris[1][35][3],iris[1][36][3],iris[1][37][3],iris[1][38][3],iris[1][39][3],iris[1][40][3],iris[1][41][3],iris[1][42][3],iris[1][43][3],iris[1][44][3],iris[1][45][3],iris[1][46][3],iris[1][47][3],iris[1][48][3],iris[1][49][3])
#print(f"Versicolor Petal Widths are: {versicolor_petal_width}")
virginica_petal_width=(iris[2][0][3],iris[2][1][3],iris[2][2][3],iris[2][3][3],iris[2][4][3],iris[2][5][3],iris[2][6][3],iris[2][7][3],iris[2][8][3],iris[2][9][3],iris[2][10][3],iris[2][11][3],iris[2][12][3],iris[2][13][3],iris[2][14][3],iris[2][15][3],iris[2][16][3],iris[2][17][3],iris[2][18][3],iris[2][19][3],iris[2][20][3],iris[2][21][3],iris[2][22][3],iris[2][23][3],iris[2][24][3],iris[2][25][3],iris[2][26][3],iris[2][27][3],iris[2][28][3],iris[2][29][3],iris[2][30][3],iris[2][31][3],iris[2][32][3],iris[2][33][3],iris[2][34][3],iris[2][35][3],iris[2][36][3],iris[2][37][3],iris[2][38][3],iris[2][39][3],iris[2][40][3],iris[2][41][3],iris[2][42][3],iris[2][43][3],iris[2][44][3],iris[2][45][3],iris[2][46][3],iris[2][47][3],iris[2][48][3],iris[2][49][3])
#print(f"Virginica Petal Widths are: {virginica_petal_width}")

fig, ax = plt.subplots()
ax.hist(setosa_petal_width, bins=10, edgecolor="black", label="Setosa", alpha=0.5)
ax.hist(versicolor_petal_width, bins=10, edgecolor="black", label="Versicolor", alpha=0.5)
ax.hist(virginica_petal_width, bins=10, edgecolor="black", label="Virginica", alpha=0.5)
ax.legend()
ax.set_xlabel("Width of Petal (cm)")
ax.set_ylabel("Frequency")
plt.title("Petal Width")
#plt.show()

# Descriptive Statistics
print(f"\nPetal Widths")
print(f"\t\tSetosa\t\tVersicolor\tVirginica")
print(f"Mean:\t\t{round(np.mean(setosa_petal_width),2)}\t\t{round(np.mean(versicolor_petal_width),2)}\t\t{round(np.mean(virginica_petal_width),2)}")
print(f"St.Dev:\t\t{round(np.std(setosa_petal_width),2)}\t\t{round(np.std(versicolor_petal_width),2)}\t\t{round(np.std(virginica_petal_width),2)}")
print(f"Median:\t\t{round(np.median(setosa_petal_width),2)}\t\t{round(np.median(versicolor_petal_width),2)}\t\t{round(np.median(virginica_petal_width),2)}")
print(f"Maximum:\t{round(np.max(setosa_petal_width),2)}\t\t{round(np.max(versicolor_petal_width),2)}\t\t{round(np.max(virginica_petal_width),2)}")
print(f"Minimum:\t{round(np.min(setosa_petal_width),2)}\t\t{round(np.min(versicolor_petal_width),2)}\t\t{round(np.min(virginica_petal_width),2)}")


setosa_petal_width=(iris[0][0][3],iris[0][1][3],iris[0][2][3],iris[0][3][3],iris[0][4][3],iris[0][5][3],iris[0][6][3],iris[0][7][3],iris[0][8][3],iris[0][9][3],iris[0][10][3],iris[0][11][3],iris[0][12][3],iris[0][13][3],iris[0][14][3],iris[0][15][3],iris[0][16][3],iris[0][17][3],iris[0][18][3],iris[0][19][3],iris[0][20][3],iris[0][21][3],iris[0][22][3],iris[0][23][3],iris[0][24][3],iris[0][25][3],iris[0][26][3],iris[0][27][3],iris[0][28][3],iris[0][29][3],iris[0][30][3],iris[0][31][3],iris[0][32][3],iris[0][33][3],iris[0][34][3],iris[0][35][3],iris[0][36][3],iris[0][37][3],iris[0][38][3],iris[0][39][3],iris[0][40][3],iris[0][41][3],iris[0][42][3],iris[0][43][3],iris[0][44][3],iris[0][45][3],iris[0][46][3],iris[0][47][3],iris[0][48][3],iris[0][49][3])
versicolor_petal_width=(iris[1][0][3],iris[1][1][3],iris[1][2][3],iris[1][3][3],iris[1][4][3],iris[1][5][3],iris[1][6][3],iris[1][7][3],iris[1][8][3],iris[1][9][3],iris[1][10][3],iris[1][11][3],iris[1][12][3],iris[1][13][3],iris[1][14][3],iris[1][15][3],iris[1][16][3],iris[1][17][3],iris[1][18][3],iris[1][19][3],iris[1][20][3],iris[1][21][3],iris[1][22][3],iris[1][23][3],iris[1][24][3],iris[1][25][3],iris[1][26][3],iris[1][27][3],iris[1][28][3],iris[1][29][3],iris[1][30][3],iris[1][31][3],iris[1][32][3],iris[1][33][3],iris[1][34][3],iris[1][35][3],iris[1][36][3],iris[1][37][3],iris[1][38][3],iris[1][39][3],iris[1][40][3],iris[1][41][3],iris[1][42][3],iris[1][43][3],iris[1][44][3],iris[1][45][3],iris[1][46][3],iris[1][47][3],iris[1][48][3],iris[1][49][3])
virginica_petal_width=(iris[2][0][3],iris[2][1][3],iris[2][2][3],iris[2][3][3],iris[2][4][3],iris[2][5][3],iris[2][6][3],iris[2][7][3],iris[2][8][3],iris[2][9][3],iris[2][10][3],iris[2][11][3],iris[2][12][3],iris[2][13][3],iris[2][14][3],iris[2][15][3],iris[2][16][3],iris[2][17][3],iris[2][18][3],iris[2][19][3],iris[2][20][3],iris[2][21][3],iris[2][22][3],iris[2][23][3],iris[2][24][3],iris[2][25][3],iris[2][26][3],iris[2][27][3],iris[2][28][3],iris[2][29][3],iris[2][30][3],iris[2][31][3],iris[2][32][3],iris[2][33][3],iris[2][34][3],iris[2][35][3],iris[2][36][3],iris[2][37][3],iris[2][38][3],iris[2][39][3],iris[2][40][3],iris[2][41][3],iris[2][42][3],iris[2][43][3],iris[2][44][3],iris[2][45][3],iris[2][46][3],iris[2][47][3],iris[2][48][3],iris[2][49][3])

petal_widths=(setosa_petal_width, versicolor_petal_width,virginica_petal_width)

setosa_petal_length=(iris[0][0][2],iris[0][1][2],iris[0][2][2],iris[0][3][2],iris[0][4][2],iris[0][5][2],iris[0][6][2],iris[0][7][2],iris[0][8][2],iris[0][9][2],iris[0][10][2],iris[0][11][2],iris[0][12][2],iris[0][13][2],iris[0][14][2],iris[0][15][2],iris[0][16][2],iris[0][17][2],iris[0][18][2],iris[0][19][2],iris[0][20][2],iris[0][21][2],iris[0][22][2],iris[0][23][2],iris[0][24][2],iris[0][25][2],iris[0][26][2],iris[0][27][2],iris[0][28][2],iris[0][29][2],iris[0][30][2],iris[0][31][2],iris[0][32][2],iris[0][33][2],iris[0][34][2],iris[0][35][2],iris[0][36][2],iris[0][37][2],iris[0][38][2],iris[0][39][2],iris[0][40][2],iris[0][41][2],iris[0][42][2],iris[0][43][2],iris[0][44][2],iris[0][45][2],iris[0][46][2],iris[0][47][2],iris[0][48][2],iris[0][49][2])
versicolor_petal_length=(iris[1][0][2],iris[1][1][2],iris[1][2][2],iris[1][3][2],iris[1][4][2],iris[1][5][2],iris[1][6][2],iris[1][7][2],iris[1][8][2],iris[1][9][2],iris[1][10][2],iris[1][11][2],iris[1][12][2],iris[1][13][2],iris[1][14][2],iris[1][15][2],iris[1][16][2],iris[1][17][2],iris[1][18][2],iris[1][19][2],iris[1][20][2],iris[1][21][2],iris[1][22][2],iris[1][23][2],iris[1][24][2],iris[1][25][2],iris[1][26][2],iris[1][27][2],iris[1][28][2],iris[1][29][2],iris[1][30][2],iris[1][31][2],iris[1][32][2],iris[1][33][2],iris[1][34][2],iris[1][35][2],iris[1][36][2],iris[1][37][2],iris[1][38][2],iris[1][39][2],iris[1][40][2],iris[1][41][2],iris[1][42][2],iris[1][43][2],iris[1][44][2],iris[1][45][2],iris[1][46][2],iris[1][47][2],iris[1][48][2],iris[1][49][2])
virginica_petal_length=(iris[2][0][2],iris[2][1][2],iris[2][2][2],iris[2][3][2],iris[2][4][2],iris[2][5][2],iris[2][6][2],iris[2][7][2],iris[2][8][2],iris[2][9][2],iris[2][10][2],iris[2][11][2],iris[2][12][2],iris[2][13][2],iris[2][14][2],iris[2][15][2],iris[2][16][2],iris[2][17][2],iris[2][18][2],iris[2][19][2],iris[2][20][2],iris[2][21][2],iris[2][22][2],iris[2][23][2],iris[2][24][2],iris[2][25][2],iris[2][26][2],iris[2][27][2],iris[2][28][2],iris[2][29][2],iris[2][30][2],iris[2][31][2],iris[2][32][2],iris[2][33][2],iris[2][34][2],iris[2][35][2],iris[2][36][2],iris[2][37][2],iris[2][38][2],iris[2][39][2],iris[2][40][2],iris[2][41][2],iris[2][42][2],iris[2][43][2],iris[2][44][2],iris[2][45][2],iris[2][46][2],iris[2][47][2],iris[2][48][2],iris[2][49][2])

petal_lengths=(setosa_petal_length, versicolor_petal_length,virginica_petal_length)

fig, ax = plt.subplots()

#Scatter plot
ax.scatter(setosa_petal_width,setosa_petal_length, marker="o",label="Setosa")
ax.scatter(versicolor_petal_width,versicolor_petal_length,marker="d", label="Versicolor")
ax.scatter(virginica_petal_width,virginica_petal_length,marker="v", label="Virginica")

# Labels
ax.set_xlabel("Petal Widths")
ax.set_ylabel("Petal Lengths")
ax.legend()

plt.title("Relationship between Petal Widths & Lengths")
#plt.show()


# Use polyfit to fit a line to the data.
m1, c1 = np.polyfit(setosa_petal_width, setosa_petal_length, 1)
m2, c2 = np.polyfit(versicolor_petal_width, versicolor_petal_length, 1)
m3, c3 = np.polyfit(virginica_petal_width, virginica_petal_length, 1)