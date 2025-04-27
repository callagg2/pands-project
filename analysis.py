# This programs reads in the data from the iris dataset
# it outputs a summary of each variable
# outputs a scatter plot of each pair of variables
# performs a linear regression to confirm the relationship between any two variables

# i have the iris data in a data subfolder

# author: gerry callaghan

# I will import matplotlib so I can use this library for statisics, plotting etc
import matplotlib.pyplot as plt
# I will import numpy so I can perform various mathematical operations with my arrays 
import numpy as np
# I will import pandas because it is great for with dataframes
import pandas as pd
# sklearn is a popular machine learning library that provides tools for data preprocessing, model selection, and evaluation.
import scipy as sp
# Import seaborn to create pairplots
import seaborn as sns

# ************************************************************************************************************
# ******************************* Verifying the data *********************************************************
# ************************************************************************************************************

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
# I'll set up some column names in accordance with the Atrribute information
col_names= ("sepal_length", "sepal_width", "petal_length", "petal_width", "class")
# create a dataframe (df) and set it equal to our imported data from the this file iris.csv
# the data is column separated, the csv file currently has no headers, so I'll add the column names as we import the data
iris_df = pd.read_csv(csv_filename, sep=',',  header= None, names=col_names)

       

print(f"The shape of the iris dataset is as follows: {iris_df.shape}\n")

# the print function has showed me there are 150 rows of data x 5 columns, 

# Eyeballing the data, I can see it has imported okay
#'''
#print(f"{iris_df}") 
#'''
# It can be seen that the first four columns are numeric data comprising the sepal/petal length/width 
# and then there is a column of alphabetical data in the iris class
# More importantly, it can be seen that everything has been sorted by the class column, so no need to sort again

# I'll use drop na to get rid of the values in the series that have no value
    # this actually returns a numpy.ndarray
iris_df.dropna(inplace=True)
# Looking at the data again to see if there was any change
#'''
#print(f"{iris_df}") 
#'''
# everything looks fine for me to work with the data now

# Just to confirm the number of different classes of iris, 
# I will use the numpy unique function on the column titled "class"

print(f"There are {iris_df["class"].nunique()} classes of iris.\n")

# so there are 3 types, which matches with the attribute information in the data file 
# which says there are Iris Setosa, Iris Versicolour, and Iris Virginica
# I am more confident that there were no typos in the class names

# Just want to doublecheck the number of instances of class
print(f"There are {iris_df["class"].size} observations.\n") # this tells there are 150 class observations

#'''
# According to the file "iris.names" which was in the zip file, we should get the following
# 5. Number of Instances: 150 (50 in each of three classes)

# I confirm this by using a quick while command (I could perhaps just create a function and call the function for each iris class )
count_setosa = 0
count_versicolor = 0
count_virginica = 0
max_rows=iris_df["class"].size

line_number = 0

while line_number < max_rows:
    if iris_df["class"][line_number] == "Iris-setosa":
        count_setosa = count_setosa + 1
        line_number = line_number + 1
    elif iris_df["class"][line_number] == "Iris-versicolor":
        count_versicolor = count_versicolor + 1 
        line_number = line_number + 1
    elif iris_df["class"][line_number] == "Iris-virginica":
        count_virginica = count_virginica + 1
        line_number = line_number + 1
    else :
        print(f"You have a typo in line {line_number}")
        line_number = line_number + 1
 
print (f"Class of Iris\tObservations \nSetosa:\t\t{count_setosa}\nVersicolor:\t{count_versicolor} \nVirginica:\t{count_virginica}\n")
#'''
# so now I know there are 150 observations, 5 columns of which 4 are numeric and one alphabetical, 
# 3 classes and definitely 50 instances of each class, which confirms the iris.names.data file

# *************************************************************************************************************************
# ************************************* Looking at the Central Tendancies *************************************************
# *************************************************************************************************************************

summary= iris_df.describe()
print(f"The central tendencies for the iris dataset is as follows: \n\n{summary}\n")

# Given that Ronald Fisher in his classic 1936 paper, "The Use of Multiple Measurements in Taxonomic Problems" was trying to use this data 
# to determine that it was possible to distinguish between the classes based on their characteristics (discriminant analysis), 
# I'm going to separate each class into its array, and then look at the characteristics to show how they differ.

# A function for creating arrays for each class of iris
def creating_array(DataFrame,min,max):
    row_number = min # starting row of the dataframe for the relevant iris class
    output_array = [] # initilise the final array as being blank
    
    while row_number < max: # while the row number of the dataframe is less than the last row of data for that respective iris class (0-50,50-100,100-150)
         current_row_of_data = iris_df["sepal_length"][row_number],iris_df["sepal_width"][row_number],iris_df["petal_length"][row_number],iris_df["petal_width"][row_number]
         row_number = row_number + 1
         output_array = output_array + [current_row_of_data]
    return output_array

# Now I'm going to call the above function to create an array for each class, 
# This function "creating_array" to incrementally append each line of the spreadsheet to the existing array
# I'm going to do what i would do in Visual Basic and do it long hand because the append function won't work for me :-( 

min=0
max_setosa = count_setosa     # above I calculated that Setosa had 50 rows, remembering the fence post rule - python read to the number BEFORE the last
setosa = np.array(creating_array(iris_df, min=0,max=max_setosa)) # so to the function, we send the dataframe, the starting point and the last row for setosa which is 50
max_versicolor = max_setosa + count_versicolor # so now we know versicolor starts after setosa, so the last row must be where count setosa + count versicolor combined, ie 100
versicolor = np.array(creating_array(iris_df, min=max_setosa,max=max_versicolor)) # so to the function, we send the dataframe, the starting point and the end point for versicolor
max_virginica = max_versicolor + count_virginica # so now we know virginica starts after versicolor, so the last row must be the previous two combined + count versicolor combined, ie 150
virginica = np.array(creating_array(iris_df, min=max_versicolor,max=max_virginica)) # so to the function, we send the dataframe, the starting point and the end point for virginica

# we can compare our each of our arrays against that of source spreadsheet
#print(f"The array for Setosa is:\n{setosa}\n\nThe array for Versicolor is:\n{versicolor}\n\nThe array for Virginica is:\n{virginica}")
# the rows all look good for each class

# Let's now create an array for iris, which in turn comprises the three arrays for setosa, versicolor and virginica
# iris = [setosa,versicolor,virginica]
# print(f"This is what an overall array comprising of 150 observations looks like: {iris}")
# okay, this matches the source data

# We want to output a summary of each variable 
# where I'm assuming each variable means feature "sepal_length", "sepal_width", "petal_length", and "petal_width"

# **********************************************************************************************************************************
# ********************** Descriptive Statistics for each of our Four Features are as follows: **************************************
# **********************************************************************************************************************************

# Sepal Lengths
# Take each of the numpy arrays we just calculated, then state the row number (between 0-150) and column number (between 0-4)
setosa_sepal_length=np.array([setosa[0][0],setosa[1][0],setosa[2][0],setosa[3][0],setosa[4][0],setosa[5][0],setosa[6][0],setosa[7][0],setosa[8][0],setosa[9][0],setosa[10][0],setosa[11][0],setosa[12][0],setosa[13][0],setosa[14][0],setosa[15][0],setosa[16][0],setosa[17][0],setosa[18][0],setosa[19][0],setosa[20][0],setosa[21][0],setosa[22][0],setosa[23][0],setosa[24][0],setosa[25][0],setosa[26][0],setosa[27][0],setosa[28][0],setosa[29][0],setosa[30][0],setosa[31][0],setosa[32][0],setosa[33][0],setosa[34][0],setosa[35][0],setosa[36][0],setosa[37][0],setosa[38][0],setosa[39][0],setosa[40][0],setosa[41][0],setosa[42][0],setosa[43][0],setosa[44][0],setosa[45][0],setosa[46][0],setosa[47][0],setosa[48][0],setosa[49][0]])
print(f"Setosa Sepal Lengths are:\n {setosa_sepal_length}\n")
versicolor_sepal_length=np.array([versicolor[0][0],versicolor[1][0],versicolor[2][0],versicolor[3][0],versicolor[4][0],versicolor[5][0],versicolor[6][0],versicolor[7][0],versicolor[8][0],versicolor[9][0],versicolor[10][0],versicolor[11][0],versicolor[12][0],versicolor[13][0],versicolor[14][0],versicolor[15][0],versicolor[16][0],versicolor[17][0],versicolor[18][0],versicolor[19][0],versicolor[20][0],versicolor[21][0],versicolor[22][0],versicolor[23][0],versicolor[24][0],versicolor[25][0],versicolor[26][0],versicolor[27][0],versicolor[28][0],versicolor[29][0],versicolor[30][0],versicolor[31][0],versicolor[32][0],versicolor[33][0],versicolor[34][0],versicolor[35][0],versicolor[36][0],versicolor[37][0],versicolor[38][0],versicolor[39][0],versicolor[40][0],versicolor[41][0],versicolor[42][0],versicolor[43][0],versicolor[44][0],versicolor[45][0],versicolor[46][0],versicolor[47][0],versicolor[48][0],versicolor[49][0]])
print(f"Versicolor Sepal Lengths are:\n {versicolor_sepal_length}\n")
virginica_sepal_length=np.array([virginica[0][0],virginica[1][0],virginica[2][0],virginica[3][0],virginica[4][0],virginica[5][0],virginica[6][0],virginica[7][0],virginica[8][0],virginica[9][0],virginica[10][0],virginica[11][0],virginica[12][0],virginica[13][0],virginica[14][0],virginica[15][0],virginica[16][0],virginica[17][0],virginica[18][0],virginica[19][0],virginica[20][0],virginica[21][0],virginica[22][0],virginica[23][0],virginica[24][0],virginica[25][0],virginica[26][0],virginica[27][0],virginica[28][0],virginica[29][0],virginica[30][0],virginica[31][0],virginica[32][0],virginica[33][0],virginica[34][0],virginica[35][0],virginica[36][0],virginica[37][0],virginica[38][0],virginica[39][0],virginica[40][0],virginica[41][0],virginica[42][0],virginica[43][0],virginica[44][0],virginica[45][0],virginica[46][0],virginica[47][0],virginica[48][0],virginica[49][0]])
print(f"Virginica Sepal Lengths are:\n {virginica_sepal_length}")
overall_sepal_length= iris_df["sepal_length"]

# Sepal Widths
# Take each of the numpy arrays we just calculated, then state the row number (between 0-150) and column number (between 0-4)
setosa_sepal_width=np.array([setosa[0][1],setosa[1][1],setosa[2][1],setosa[3][1],setosa[4][1],setosa[5][1],setosa[6][1],setosa[7][1],setosa[8][1],setosa[9][1],setosa[10][1],setosa[11][1],setosa[12][1],setosa[13][1],setosa[14][1],setosa[15][1],setosa[16][1],setosa[17][1],setosa[18][1],setosa[19][1],setosa[20][1],setosa[21][1],setosa[22][1],setosa[23][1],setosa[24][1],setosa[25][1],setosa[26][1],setosa[27][1],setosa[28][1],setosa[29][1],setosa[30][1],setosa[31][1],setosa[32][1],setosa[33][1],setosa[34][1],setosa[35][1],setosa[36][1],setosa[37][1],setosa[38][1],setosa[39][1],setosa[40][1],setosa[41][1],setosa[42][1],setosa[43][1],setosa[44][1],setosa[45][1],setosa[46][1],setosa[47][1],setosa[48][1],setosa[49][1]])
print(f"\nSetosa Sepal widths are:\n {setosa_sepal_width}\n")
versicolor_sepal_width=np.array([versicolor[0][1],versicolor[1][1],versicolor[2][1],versicolor[3][1],versicolor[4][1],versicolor[5][1],versicolor[6][1],versicolor[7][1],versicolor[8][1],versicolor[9][1],versicolor[10][1],versicolor[11][1],versicolor[12][1],versicolor[13][1],versicolor[14][1],versicolor[15][1],versicolor[16][1],versicolor[17][1],versicolor[18][1],versicolor[19][1],versicolor[20][1],versicolor[21][1],versicolor[22][1],versicolor[23][1],versicolor[24][1],versicolor[25][1],versicolor[26][1],versicolor[27][1],versicolor[28][1],versicolor[29][1],versicolor[30][1],versicolor[31][1],versicolor[32][1],versicolor[33][1],versicolor[34][1],versicolor[35][1],versicolor[36][1],versicolor[37][1],versicolor[38][1],versicolor[39][1],versicolor[40][1],versicolor[41][1],versicolor[42][1],versicolor[43][1],versicolor[44][1],versicolor[45][1],versicolor[46][1],versicolor[47][1],versicolor[48][1],versicolor[49][1]])
print(f"Versicolor Sepal widths are:\n {versicolor_sepal_width}\n")
virginica_sepal_width=np.array([virginica[0][1],virginica[1][1],virginica[2][1],virginica[3][1],virginica[4][1],virginica[5][1],virginica[6][1],virginica[7][1],virginica[8][1],virginica[9][1],virginica[10][1],virginica[11][1],virginica[12][1],virginica[13][1],virginica[14][1],virginica[15][1],virginica[16][1],virginica[17][1],virginica[18][1],virginica[19][1],virginica[20][1],virginica[21][1],virginica[22][1],virginica[23][1],virginica[24][1],virginica[25][1],virginica[26][1],virginica[27][1],virginica[28][1],virginica[29][1],virginica[30][1],virginica[31][1],virginica[32][1],virginica[33][1],virginica[34][1],virginica[35][1],virginica[36][1],virginica[37][1],virginica[38][1],virginica[39][1],virginica[40][1],virginica[41][1],virginica[42][1],virginica[43][1],virginica[44][1],virginica[45][1],virginica[46][1],virginica[47][1],virginica[48][1],virginica[49][1]])
print(f"Virginica Sepal widths are:\n {virginica_sepal_width}")
overall_sepal_width= iris_df["sepal_width"]

# Petal Lengths
# Take each of the numpy arrays we just calculated, then state the row number (between 0-150) and column number (between 0-4)
setosa_petal_length=np.array([setosa[0][2],setosa[1][2],setosa[2][2],setosa[3][2],setosa[4][2],setosa[5][2],setosa[6][2],setosa[7][2],setosa[8][2],setosa[9][2],setosa[10][2],setosa[11][2],setosa[12][2],setosa[13][2],setosa[14][2],setosa[15][2],setosa[16][2],setosa[17][2],setosa[18][2],setosa[19][2],setosa[20][2],setosa[21][2],setosa[22][2],setosa[23][2],setosa[24][2],setosa[25][2],setosa[26][2],setosa[27][2],setosa[28][2],setosa[29][2],setosa[30][2],setosa[31][2],setosa[32][2],setosa[33][2],setosa[34][2],setosa[35][2],setosa[36][2],setosa[37][2],setosa[38][2],setosa[39][2],setosa[40][2],setosa[41][2],setosa[42][2],setosa[43][2],setosa[44][2],setosa[45][2],setosa[46][2],setosa[47][2],setosa[48][2],setosa[49][2]])
print(f"\nSetosa Petal Lengths are:\n {setosa_petal_length}\n")
versicolor_petal_length=np.array([versicolor[0][2],versicolor[1][2],versicolor[2][2],versicolor[3][2],versicolor[4][2],versicolor[5][2],versicolor[6][2],versicolor[7][2],versicolor[8][2],versicolor[9][2],versicolor[10][2],versicolor[11][2],versicolor[12][2],versicolor[13][2],versicolor[14][2],versicolor[15][2],versicolor[16][2],versicolor[17][2],versicolor[18][2],versicolor[19][2],versicolor[20][2],versicolor[21][2],versicolor[22][2],versicolor[23][2],versicolor[24][2],versicolor[25][2],versicolor[26][2],versicolor[27][2],versicolor[28][2],versicolor[29][2],versicolor[30][2],versicolor[31][2],versicolor[32][2],versicolor[33][2],versicolor[34][2],versicolor[35][2],versicolor[36][2],versicolor[37][2],versicolor[38][2],versicolor[39][2],versicolor[40][2],versicolor[41][2],versicolor[42][2],versicolor[43][2],versicolor[44][2],versicolor[45][2],versicolor[46][2],versicolor[47][2],versicolor[48][2],versicolor[49][2]])
print(f"Versicolor Petal Lengths are:\n {versicolor_petal_length}\n")
virginica_petal_length=np.array([virginica[0][2],virginica[1][2],virginica[2][2],virginica[3][2],virginica[4][2],virginica[5][2],virginica[6][2],virginica[7][2],virginica[8][2],virginica[9][2],virginica[10][2],virginica[11][2],virginica[12][2],virginica[13][2],virginica[14][2],virginica[15][2],virginica[16][2],virginica[17][2],virginica[18][2],virginica[19][2],virginica[20][2],virginica[21][2],virginica[22][2],virginica[23][2],virginica[24][2],virginica[25][2],virginica[26][2],virginica[27][2],virginica[28][2],virginica[29][2],virginica[30][2],virginica[31][2],virginica[32][2],virginica[33][2],virginica[34][2],virginica[35][2],virginica[36][2],virginica[37][2],virginica[38][2],virginica[39][2],virginica[40][2],virginica[41][2],virginica[42][2],virginica[43][2],virginica[44][2],virginica[45][2],virginica[46][2],virginica[47][2],virginica[48][2],virginica[49][2]])
print(f"Virginica Petal Lengths are:\n {virginica_petal_length}")
overall_petal_length=iris_df["petal_length"]

# Petal Widths
# Take each of the numpy arrays we just calculated, then state the row number (between 0-150) and column number (between 0-4)
setosa_petal_width=np.array([setosa[0][3],setosa[1][3],setosa[2][3],setosa[3][3],setosa[4][3],setosa[5][3],setosa[6][3],setosa[7][3],setosa[8][3],setosa[9][3],setosa[10][3],setosa[11][3],setosa[12][3],setosa[13][3],setosa[14][3],setosa[15][3],setosa[16][3],setosa[17][3],setosa[18][3],setosa[19][3],setosa[20][3],setosa[21][3],setosa[22][3],setosa[23][3],setosa[24][3],setosa[25][3],setosa[26][3],setosa[27][3],setosa[28][3],setosa[29][3],setosa[30][3],setosa[31][3],setosa[32][3],setosa[33][3],setosa[34][3],setosa[35][3],setosa[36][3],setosa[37][3],setosa[38][3],setosa[39][3],setosa[40][3],setosa[41][3],setosa[42][3],setosa[43][3],setosa[44][3],setosa[45][3],setosa[46][3],setosa[47][3],setosa[48][3],setosa[49][3]])
print(f"\nSetosa Petal Widths are:\n {setosa_petal_width}\n")
versicolor_petal_width=np.array([versicolor[0][3],versicolor[1][3],versicolor[2][3],versicolor[3][3],versicolor[4][3],versicolor[5][3],versicolor[6][3],versicolor[7][3],versicolor[8][3],versicolor[9][3],versicolor[10][3],versicolor[11][3],versicolor[12][3],versicolor[13][3],versicolor[14][3],versicolor[15][3],versicolor[16][3],versicolor[17][3],versicolor[18][3],versicolor[19][3],versicolor[20][3],versicolor[21][3],versicolor[22][3],versicolor[23][3],versicolor[24][3],versicolor[25][3],versicolor[26][3],versicolor[27][3],versicolor[28][3],versicolor[29][3],versicolor[30][3],versicolor[31][3],versicolor[32][3],versicolor[33][3],versicolor[34][3],versicolor[35][3],versicolor[36][3],versicolor[37][3],versicolor[38][3],versicolor[39][3],versicolor[40][3],versicolor[41][3],versicolor[42][3],versicolor[43][3],versicolor[44][3],versicolor[45][3],versicolor[46][3],versicolor[47][3],versicolor[48][3],versicolor[49][3]])
print(f"Versicolor Petal Widths are:\n {versicolor_petal_width}\n")
virginica_petal_width=np.array([virginica[0][3],virginica[1][3],virginica[2][3],virginica[3][3],virginica[4][3],virginica[5][3],virginica[6][3],virginica[7][3],virginica[8][3],virginica[9][3],virginica[10][3],virginica[11][3],virginica[12][3],virginica[13][3],virginica[14][3],virginica[15][3],virginica[16][3],virginica[17][3],virginica[18][3],virginica[19][3],virginica[20][3],virginica[21][3],virginica[22][3],virginica[23][3],virginica[24][3],virginica[25][3],virginica[26][3],virginica[27][3],virginica[28][3],virginica[29][3],virginica[30][3],virginica[31][3],virginica[32][3],virginica[33][3],virginica[34][3],virginica[35][3],virginica[36][3],virginica[37][3],virginica[38][3],virginica[39][3],virginica[40][3],virginica[41][3],virginica[42][3],virginica[43][3],virginica[44][3],virginica[45][3],virginica[46][3],virginica[47][3],virginica[48][3],virginica[49][3]])
print(f"Virginica Petal Widths are:\n {virginica_petal_width}")
overall_petal_width=iris_df["petal_width"]


# ************************************* Printing the Summary Statistics ************************************************************

# Sepal Lengths
# Descriptive Statistics
print(f"\n\t\t\tCentral Tendencies")
print(f"\nSepal Lengths")
print(f"\t\tSetosa\t\tVersicolor\tVirginica\tOverall")
print(f"Count:\t\t{count_setosa}\t\t{count_versicolor}\t\t{count_virginica}\t\t{max_rows}")
print(f"Mean:\t\t{round(np.mean(setosa_sepal_length),2)}\t\t{round(np.mean(versicolor_sepal_length),2)}\t\t{round(np.mean(virginica_sepal_length),2)}\t\t{round(np.mean(overall_sepal_length),2)}")
print(f"St.Dev:\t\t{round(np.std(setosa_sepal_length),2)}\t\t{round(np.std(versicolor_sepal_length),2)}\t\t{round(np.std(virginica_sepal_length),2)}\t\t{round(np.std(overall_sepal_length),2)}")
print(f"Median:\t\t{round(np.median(setosa_sepal_length),2)}\t\t{round(np.median(versicolor_sepal_length),2)}\t\t{round(np.median(virginica_sepal_length),2)}\t\t{round(np.median(overall_sepal_length),2)}")
print(f"Minimum:\t{round(np.min(setosa_sepal_length),2)}\t\t{round(np.min(versicolor_sepal_length),2)}\t\t{round(np.min(virginica_sepal_length),2)}\t\t{round(np.min(overall_sepal_length),2)}")
print(f"25%:\t\t{round(np.percentile(setosa_sepal_length,25),2)}\t\t{round(np.percentile(versicolor_sepal_length,25),2)}\t\t{round(np.percentile(virginica_sepal_length,25),2)}\t\t{round(np.percentile(overall_sepal_length,25),2)}")
print(f"50%:\t\t{round(np.percentile(setosa_sepal_length,50),2)}\t\t{round(np.percentile(versicolor_sepal_length,50),2)}\t\t{round(np.percentile(virginica_sepal_length,50),2)}\t\t{round(np.percentile(overall_sepal_length,50),2)}")
print(f"75%:\t\t{round(np.percentile(setosa_sepal_length,75),2)}\t\t{round(np.percentile(versicolor_sepal_length,75),2)}\t\t{round(np.percentile(virginica_sepal_length,75),2)}\t\t{round(np.percentile(overall_sepal_length,75),2)}")
print(f"Maximum:\t{round(np.max(setosa_sepal_length),2)}\t\t{round(np.max(versicolor_sepal_length),2)}\t\t{round(np.max(virginica_sepal_length),2)}\t\t{round(np.max(overall_sepal_length),2)}")

# Sepal Widths
# Descriptive Statistics
print(f"\nSepal Widths")
print(f"\t\tSetosa\t\tVersicolor\tVirginica\tOverall")
print(f"Count:\t\t{count_setosa}\t\t{count_versicolor}\t\t{count_virginica}\t\t{max_rows}")
print(f"Mean:\t\t{round(np.mean(setosa_sepal_width),2)}\t\t{round(np.mean(versicolor_sepal_width),2)}\t\t{round(np.mean(virginica_sepal_width),2)}\t\t{round(np.mean(overall_sepal_width),2)}")
print(f"St.Dev:\t\t{round(np.std(setosa_sepal_width),2)}\t\t{round(np.std(versicolor_sepal_width),2)}\t\t{round(np.std(virginica_sepal_width),2)}\t\t{round(np.std(overall_sepal_width),2)}")
print(f"Median:\t\t{round(np.median(setosa_sepal_width),2)}\t\t{round(np.median(versicolor_sepal_width),2)}\t\t{round(np.median(virginica_sepal_width),2)}\t\t{round(np.median(overall_sepal_width),2)}")
print(f"Minimum:\t{round(np.min(setosa_sepal_width),2)}\t\t{round(np.min(versicolor_sepal_width),2)}\t\t{round(np.min(virginica_sepal_width),2)}\t\t{round(np.min(overall_sepal_width),2)}")
print(f"25%:\t\t{round(np.percentile(setosa_sepal_width,25),2)}\t\t{round(np.percentile(versicolor_sepal_width,25),2)}\t\t{round(np.percentile(virginica_sepal_width,25),2)}\t\t{round(np.percentile(overall_sepal_width,25),2)}")
print(f"50%:\t\t{round(np.percentile(setosa_sepal_width,50),2)}\t\t{round(np.percentile(versicolor_sepal_width,50),2)}\t\t{round(np.percentile(virginica_sepal_width,50),2)}\t\t{round(np.percentile(overall_sepal_width,50),2)}")
print(f"75%:\t\t{round(np.percentile(setosa_sepal_width,75),2)}\t\t{round(np.percentile(versicolor_sepal_width,75),2)}\t\t{round(np.percentile(virginica_sepal_width,75),2)}\t\t{round(np.percentile(overall_sepal_width,75),2)}")
print(f"Maximum:\t{round(np.max(setosa_sepal_width),2)}\t\t{round(np.max(versicolor_sepal_width),2)}\t\t{round(np.max(virginica_sepal_width),2)}\t\t{round(np.max(overall_sepal_width),2)}")

# Petal Lengths
# Descriptive Statistics
print(f"\nPetal Length")
print(f"\t\tSetosa\t\tVersicolor\tVirginica\tOverall")
print(f"Count:\t\t{count_setosa}\t\t{count_versicolor}\t\t{count_virginica}\t\t{max_rows}")
print(f"Mean:\t\t{round(np.mean(setosa_petal_length),2)}\t\t{round(np.mean(versicolor_petal_length),2)}\t\t{round(np.mean(virginica_petal_length),2)}\t\t{round(np.mean(overall_petal_length),2)}")
print(f"St.Dev:\t\t{round(np.std(setosa_petal_length),2)}\t\t{round(np.std(versicolor_petal_length),2)}\t\t{round(np.std(virginica_petal_length),2)}\t\t{round(np.std(overall_petal_length),2)}")
print(f"Median:\t\t{round(np.median(setosa_petal_length),2)}\t\t{round(np.median(versicolor_petal_length),2)}\t\t{round(np.median(virginica_petal_length),2)}\t\t{round(np.median(overall_petal_length),2)}")
print(f"Minimum:\t{round(np.min(setosa_petal_length),2)}\t\t{round(np.min(versicolor_petal_length),2)}\t\t{round(np.min(virginica_petal_length),2)}\t\t{round(np.min(overall_petal_length),2)}")
print(f"25%:\t\t{round(np.percentile(setosa_petal_length,25),2)}\t\t{round(np.percentile(versicolor_petal_length,25),2)}\t\t{round(np.percentile(virginica_petal_length,25),2)}\t\t{round(np.percentile(overall_petal_length,25),2)}")
print(f"50%:\t\t{round(np.percentile(setosa_petal_length,50),2)}\t\t{round(np.percentile(versicolor_petal_length,50),2)}\t\t{round(np.percentile(virginica_petal_length,50),2)}\t\t{round(np.percentile(overall_petal_length,50),2)}")
print(f"75%:\t\t{round(np.percentile(setosa_petal_length,75),2)}\t\t{round(np.percentile(versicolor_petal_length,75),2)}\t\t{round(np.percentile(virginica_petal_length,75),2)}\t\t{round(np.percentile(overall_petal_length,75),2)}")
print(f"Maximum:\t{round(np.max(setosa_petal_length),2)}\t\t{round(np.max(versicolor_petal_length),2)}\t\t{round(np.max(virginica_petal_length),2)}\t\t{round(np.max(overall_petal_length),2)}")

# Petal Widths
# Descriptive Statistics
print(f"\nPetal Widths")
print(f"\t\tSetosa\t\tVersicolor\tVirginica\tOverall")
print(f"Count:\t\t{count_setosa}\t\t{count_versicolor}\t\t{count_virginica}\t\t{max_rows}")
print(f"Mean:\t\t{round(np.mean(setosa_petal_width),2)}\t\t{round(np.mean(versicolor_petal_width),2)}\t\t{round(np.mean(virginica_petal_width),2)}\t\t{round(np.mean(overall_petal_width),2)}")
print(f"St.Dev:\t\t{round(np.std(setosa_petal_width),2)}\t\t{round(np.std(versicolor_petal_width),2)}\t\t{round(np.std(virginica_petal_width),2)}\t\t{round(np.std(overall_petal_width),2)}")
print(f"Median:\t\t{round(np.median(setosa_petal_width),2)}\t\t{round(np.median(versicolor_petal_width),2)}\t\t{round(np.median(virginica_petal_width),2)}\t\t{round(np.median(overall_petal_width),2)}")
print(f"Minimum:\t{round(np.min(setosa_petal_width),2)}\t\t{round(np.min(versicolor_petal_width),2)}\t\t{round(np.min(virginica_petal_width),2)}\t\t{round(np.min(overall_petal_width),2)}")
print(f"25%:\t\t{round(np.percentile(setosa_petal_width,25),2)}\t\t{round(np.percentile(versicolor_petal_width,25),2)}\t\t{round(np.percentile(virginica_petal_width,25),2)}\t\t{round(np.percentile(overall_petal_width,25),2)}")
print(f"50%:\t\t{round(np.percentile(setosa_petal_width,50),2)}\t\t{round(np.percentile(versicolor_petal_width,50),2)}\t\t{round(np.percentile(virginica_petal_width,50),2)}\t\t{round(np.percentile(overall_petal_width,50),2)}")
print(f"75%:\t\t{round(np.percentile(setosa_petal_width,75),2)}\t\t{round(np.percentile(versicolor_petal_width,75),2)}\t\t{round(np.percentile(virginica_petal_width,75),2)}\t\t{round(np.percentile(overall_petal_width,75),2)}")
print(f"Maximum:\t{round(np.max(setosa_petal_width),2)}\t\t{round(np.max(versicolor_petal_width),2)}\t\t{round(np.max(virginica_petal_width),2)}\t\t{round(np.max(overall_petal_width),2)}\n")

# ******************************************************************************************************************************************************

# ********************* Printing the summary information to an external text files in the same directory *****************************************

# from w3 schools Create a new file if it does not exist: f = open("myfile.txt", "w")
# so let's call our file sepal_lengths.txt
# this now prints the above information to a text file called sepal_lengths.txt in the same directory
# i have to convert every number to a string otherwise it won't write to the file

# Sepal Lengths
with open("sepal_lengths.txt","w") as f:
     f.write("sepal lengths" + "\n")
     f.write("\t"+"\t"+"Setosa" +"\t"+"Versicolor"+"\t"+"Virginica"+"\t"+"Overall" + "\n")
     f.write("Count:"+"\t"+str(count_setosa) +"\t"+"\t"+str(count_versicolor)+"\t"+"\t"+"\t"+str(count_virginica)+"\t"+"\t"+"\t"+str(max_rows) + "\n")
     f.write("Mean:"+"\t"+str(round(np.mean(setosa_sepal_length),2)) +"\t"+str(round(np.mean(versicolor_sepal_length),2)) +"\t"+"\t"+str(round(np.mean(virginica_sepal_length),2)) +"\t"+"\t"+str(round(np.mean(overall_sepal_length),2)) +"\n")
     f.write("Std.D:"+"\t"+str(round(np.std(setosa_sepal_length),2)) +"\t"+str(round(np.std(versicolor_sepal_length),2)) +"\t"+"\t"+str(round(np.std(virginica_sepal_length),2)) +"\t"+"\t"+str(round(np.std(overall_sepal_length),2)) +"\n")
     f.write("Median:"+"\t"+str(round(np.median(setosa_sepal_length),2)) +"\t"+"\t"+str(round(np.median(versicolor_sepal_length),2)) +"\t"+"\t"+"\t"+str(round(np.median(virginica_sepal_length),2)) +"\t"+"\t"+"\t"+str(round(np.median(overall_sepal_length),2)) +"\n")
     f.write("Min:"+"\t"+str(round(np.min(setosa_sepal_length),2)) +"\t"+"\t"+str(round(np.min(versicolor_sepal_length),2)) +"\t"+"\t"+"\t"+str(round(np.min(virginica_sepal_length),2)) +"\t"+"\t"+"\t"+str(round(np.min(overall_sepal_length),2)) +"\n")
     f.write("25%:"+"\t"+str(round(np.percentile(setosa_sepal_length,25),2)) +"\t"+"\t"+str(round(np.percentile(versicolor_sepal_length,25),2)) +"\t"+"\t"+"\t"+str(round(np.percentile(virginica_sepal_length,25),2)) +"\t"+"\t"+str(round(np.percentile(overall_sepal_length,25),2)) +"\n")
     f.write("50%:"+"\t"+str(round(np.percentile(setosa_sepal_length,50),2)) +"\t"+"\t"+str(round(np.percentile(versicolor_sepal_length,50),2)) +"\t"+"\t"+"\t"+str(round(np.percentile(virginica_sepal_length,50),2)) +"\t"+"\t"+"\t"+str(round(np.percentile(overall_sepal_length,50),2))+"\n" )
     f.write("75%:"+"\t"+str(round(np.percentile(setosa_sepal_length,75),2)) +"\t"+"\t"+str(round(np.percentile(versicolor_sepal_length,75),2)) +"\t"+"\t"+"\t"+str(round(np.percentile(virginica_sepal_length,75),2)) +"\t"+"\t"+"\t"+str(round(np.percentile(overall_sepal_length,75),2)) +"\n")
     f.write("Max:"+"\t"+str(round(np.max(setosa_sepal_length),2)) +"\t"+"\t"+str(round(np.max(versicolor_sepal_length),2)) +"\t"+"\t"+"\t"+str(round(np.max(virginica_sepal_length),2)) +"\t"+"\t"+"\t"+str(round(np.max(overall_sepal_length),2)) +"\n"+"\n")

# Sepal Widths
# this now prints the above information to a text file called sepal_widths.txt in the same directory
with open("sepal_widths.txt","w") as f:
     f.write("sepal widths" + "\n")
     f.write("\t"+"\t"+"Setosa" +"\t"+"Versicolor"+"\t"+"Virginica"+"\t"+"Overall" + "\n")
     f.write("Count:"+"\t"+str(count_setosa) +"\t"+"\t"+str(count_versicolor)+"\t"+"\t"+"\t"+str(count_virginica)+"\t"+"\t"+"\t"+str(max_rows) + "\n")
     f.write("Mean:"+"\t"+str(round(np.mean(setosa_sepal_width),2)) +"\t"+str(round(np.mean(versicolor_sepal_width),2)) +"\t"+"\t"+str(round(np.mean(virginica_sepal_width),2)) +"\t"+"\t"+str(round(np.mean(overall_sepal_width),2)) +"\n")
     f.write("Std.D:"+"\t"+str(round(np.std(setosa_sepal_width),2)) +"\t"+str(round(np.std(versicolor_sepal_width),2)) +"\t"+"\t"+ str(round(np.std(virginica_sepal_width),2)) +"\t"+"\t"+str(round(np.std(overall_sepal_width),2)) +"\n")
     f.write("Median:"+"\t"+str(round(np.median(setosa_sepal_width),2)) +"\t"+"\t"+str(round(np.median(versicolor_sepal_width),2))+"\t" +"\t"+"\t"+str(round(np.median(virginica_sepal_width),2)) +"\t"+"\t"+"\t"+str(round(np.median(overall_sepal_width),2)) +"\n")
     f.write("Min:"+"\t"+str(round(np.min(setosa_sepal_width),2)) +"\t"+"\t"+str(round(np.min(versicolor_sepal_width),2)) +"\t"+"\t"+"\t"+str(round(np.min(virginica_sepal_width),2)) +"\t"+"\t"+"\t"+str(round(np.min(overall_sepal_width),2)) +"\n")
     f.write("25%:"+"\t"+str(round(np.percentile(setosa_sepal_width,25),2)) +"\t"+str(round(np.percentile(versicolor_sepal_width,25),2)) +"\t"+"\t"+ str(round(np.percentile(virginica_sepal_width,25),2)) +"\t"+"\t"+"\t"+str(round(np.percentile(overall_sepal_width,25),2)) +"\n")
     f.write("50%:"+"\t"+str(round(np.percentile(setosa_sepal_width,50),2)) +"\t"+"\t"+str(round(np.percentile(versicolor_sepal_width,50),2)) +"\t"+"\t"+"\t"+str(round(np.percentile(virginica_sepal_width,50),2)) +"\t"+"\t"+"\t"+str(round(np.percentile(overall_sepal_width,50),2))+"\n" )
     f.write("75%:"+"\t"+str(round(np.percentile(setosa_sepal_width,75),2)) +"\t"+str(round(np.percentile(versicolor_sepal_width,75),2)) +"\t"+"\t"+"\t"+ str(round(np.percentile(virginica_sepal_width,75),2)) +"\t"+"\t" + str(round(np.percentile(overall_sepal_width,75),2)) +"\n")
     f.write("Max:"+"\t"+str(round(np.max(setosa_sepal_width),2)) +"\t"+"\t"+str(round(np.max(versicolor_sepal_width),2)) +"\t"+"\t"+"\t"+str(round(np.max(virginica_sepal_width),2)) +"\t"+"\t"+"\t"+str(round(np.max(overall_sepal_width),2)) +"\n"+"\n")

# Petal Lengths
# this now prints the above information to a text file called petal_lengths.txt in the same directory
with open("petal_lengths.txt","w") as f:
     f.write("Petal lengths" + "\n")
     f.write("\t"+"\t"+"Setosa" +"\t"+"Versicolor"+"\t"+"Virginica"+"\t"+"Overall" + "\n")
     f.write("Count:"+"\t"+str(count_setosa) +"\t"+"\t"+str(count_versicolor)+"\t"+"\t"+"\t"+str(count_virginica)+"\t"+"\t"+"\t"+str(max_rows) + "\n")
     f.write("Mean:"+"\t"+str(round(np.mean(setosa_petal_length),2)) +"\t"+str(round(np.mean(versicolor_petal_length),2)) +"\t"+"\t"+str(round(np.mean(virginica_petal_length),2)) +"\t"+"\t"+str(round(np.mean(overall_petal_length),2)) +"\n")
     f.write("Std.D:"+"\t"+str(round(np.std(setosa_petal_length),2)) +"\t"+str(round(np.std(versicolor_petal_length),2)) +"\t"+"\t"+ str(round(np.std(virginica_petal_length),2)) +"\t"+"\t"+str(round(np.std(overall_petal_length),2)) +"\n")
     f.write("Median:"+"\t"+str(round(np.median(setosa_petal_length),2)) +"\t"+"\t"+str(round(np.median(versicolor_petal_length),2)) +"\t"+"\t"+ str(round(np.median(virginica_petal_length),2)) +"\t"+"\t"+str(round(np.median(overall_petal_length),2)) +"\n")
     f.write("Min:"+"\t"+str(round(np.min(setosa_petal_length),2)) +"\t"+"\t"+str(round(np.min(versicolor_petal_length),2)) +"\t"+"\t"+"\t"+ str(round(np.min(virginica_petal_length),2)) +"\t"+"\t"+"\t"+str(round(np.min(overall_petal_length),2)) +"\n")
     f.write("25%:"+"\t"+str(round(np.percentile(setosa_petal_length,25),2)) +"\t"+"\t"+str(round(np.percentile(versicolor_petal_length,25),2)) +"\t"+"\t"+"\t"+str(round(np.percentile(virginica_petal_length,25),2)) +"\t"+"\t"+"\t"+str(round(np.percentile(overall_petal_length,25),2)) +"\n")
     f.write("50%:"+"\t"+str(round(np.percentile(setosa_petal_length,50),2)) +"\t"+"\t"+str(round(np.percentile(versicolor_petal_length,50),2)) +"\t"+"\t"+ str(round(np.percentile(virginica_petal_length,50),2)) +"\t"+"\t"+str(round(np.percentile(overall_petal_length,50),2))+"\n" )
     f.write("75%:"+"\t"+str(round(np.percentile(setosa_petal_length,75),2)) +"\t"+str(round(np.percentile(versicolor_petal_length,75),2)) +"\t"+"\t"+"\t"+str(round(np.percentile(virginica_petal_length,75),2)) +"\t"+"\t"+str(round(np.percentile(overall_petal_length,75),2)) +"\n")
     f.write("Max:"+"\t"+str(round(np.max(setosa_petal_length),2)) +"\t"+"\t"+str(round(np.max(versicolor_petal_length),2)) +"\t"+"\t"+"\t"+str(round(np.max(virginica_petal_length),2)) +"\t"+"\t"+"\t"+str(round(np.max(overall_petal_length),2)) +"\n")

# Petal Widths
# this now prints the above information to a text file called petal_widths.txt in the same directory
with open("petal_widths.txt","w") as f:
     f.write("Petal Widths" + "\n")
     f.write("\t"+"\t"+"Setosa" +"\t"+"Versicolor"+"\t"+"Virginica"+"\t"+"Overall" + "\n")
     f.write("Count:"+"\t"+str(count_setosa) +"\t"+"\t"+str(count_versicolor)+"\t"+"\t"+"\t"+str(count_virginica)+"\t"+"\t"+"\t"+str(max_rows) + "\n")
     f.write("Mean:"+"\t"+str(round(np.mean(setosa_petal_width),2)) +"\t"+str(round(np.mean(versicolor_petal_width),2)) +"\t"+"\t"+str(round(np.mean(virginica_petal_width),2)) +"\t"+"\t"+str(round(np.mean(overall_petal_width),2)) +"\n")
     f.write("Std.D:"+"\t"+str(round(np.std(setosa_petal_width),2)) +"\t"+str(round(np.std(versicolor_petal_width),2)) +"\t"+"\t"+"\t"+str(round(np.std(virginica_petal_width),2)) +"\t"+"\t"+str(round(np.std(overall_petal_width),2)) +"\n")
     f.write("Median:"+"\t"+str(round(np.median(setosa_petal_width),2)) +"\t"+"\t"+str(round(np.median(versicolor_petal_width),2)) +"\t"+"\t"+"\t"+str(round(np.median(virginica_petal_width),2)) +"\t"+"\t"+"\t"+str(round(np.median(overall_petal_width),2)) +"\n")
     f.write("Min:"+"\t"+str(round(np.min(setosa_petal_width),2)) +"\t"+"\t"+str(round(np.min(versicolor_petal_width),2)) +"\t"+"\t"+"\t"+str(round(np.min(virginica_petal_width),2)) +"\t"+"\t"+"\t"+str(round(np.min(overall_petal_width),2)) +"\n")
     f.write("25%:"+"\t"+str(round(np.percentile(setosa_petal_width,25),2)) +"\t"+"\t"+str(round(np.percentile(versicolor_petal_width,25),2)) +"\t"+"\t"+"\t"+str(round(np.percentile(virginica_petal_width,25),2)) +"\t"+"\t"+"\t"+str(round(np.percentile(overall_petal_width,25),2)) +"\n")
     f.write("50%:"+"\t"+str(round(np.percentile(setosa_petal_width,50),2)) +"\t"+"\t"+str(round(np.percentile(versicolor_petal_width,50),2)) +"\t"+"\t"+"\t"+str(round(np.percentile(virginica_petal_width,50),2)) +"\t"+"\t"+"\t"+str(round(np.percentile(overall_petal_width,50),2))+"\n" )
     f.write("75%:"+"\t"+str(round(np.percentile(setosa_petal_width,75),2)) +"\t"+"\t"+str(round(np.percentile(versicolor_petal_width,75),2)) +"\t"+"\t"+"\t"+str(round(np.percentile(virginica_petal_width,75),2)) +"\t"+"\t"+"\t"+str(round(np.percentile(overall_petal_width,75),2)) +"\n")
     f.write("Max:"+"\t"+str(round(np.max(setosa_petal_width),2)) +"\t"+"\t"+str(round(np.max(versicolor_petal_width),2)) +"\t"+"\t"+"\t"+str(round(np.max(virginica_petal_width),2)) +"\t"+"\t"+"\t"+str(round(np.max(overall_petal_width),2)) +"\n"+"\n")

#*************************************************************************************************************************************
# ********************* Visualizing the data on histograms, that is, plotting the data *************************************************************

# Sepal Lengths
fig, ax = plt.subplots()
# I'm going to separate each class of iris on the historgram because it can help determine how they differ
ax.hist(setosa_sepal_length, bins=10, edgecolor="black", label="Setosa", alpha=0.5)
ax.hist(versicolor_sepal_length, bins=10, edgecolor="black", label="Versicolor", alpha=0.5)
ax.hist(virginica_sepal_length, bins=10, edgecolor="black", label="Virginica", alpha=0.5)

# legend 
ax.legend()

# axis labels
ax.set_xlabel("Length of Sepal (cm)")
ax.set_ylabel("Frequency")

# title
plt.title("Sepal Lengths")

#plt.show()
plt.savefig("histogram_of_sepal_lengths.png")

# Sepal Widths
fig, ax = plt.subplots()
# I'm going to separate each class of iris on the historgram because it can help determine how they differ
ax.hist(setosa_sepal_width, bins=10, edgecolor="black", label="Setosa", alpha=0.5)
ax.hist(versicolor_sepal_width, bins=10, edgecolor="black", label="Versicolor", alpha=0.5)
ax.hist(virginica_sepal_width, bins=10, edgecolor="black", label="Virginica", alpha=0.5)

# legend
ax.legend()

# axis labels
ax.set_xlabel("Width of Sepal (cm)")
ax.set_ylabel("Frequency")

# title
plt.title("Sepal Widths")

#plt.show()
plt.savefig("histogram_of_sepal_widths.png")

# Petal Lengths
fig, ax = plt.subplots()
# I'm going to separate each class of iris on the historgram because it can help determine how they differ
ax.hist(setosa_petal_length, bins=10, edgecolor="black", label="Setosa", alpha=0.5)
ax.hist(versicolor_petal_length, bins=10, edgecolor="black", label="Versicolor", alpha=0.5)
ax.hist(virginica_petal_length, bins=10, edgecolor="black", label="Virginica", alpha=0.5)

#legend
ax.legend()

#axis titles
ax.set_xlabel("Length of Petal (cm)")
ax.set_ylabel("Frequency")

# title
plt.title("Petal Lengths")

#plt.show()
plt.savefig("histogram_of_petal_lengths.png")

# Petal Widths
fig, ax = plt.subplots()
# I'm going to separate each class of iris on the historgram because it can help determine how they differ
ax.hist(setosa_petal_width, bins=10, edgecolor="black", label="Setosa", alpha=0.5)
ax.hist(versicolor_petal_width, bins=10, edgecolor="black", label="Versicolor", alpha=0.5)
ax.hist(virginica_petal_width, bins=10, edgecolor="black", label="Virginica", alpha=0.5)

# legend
ax.legend()

# axis titles
ax.set_xlabel("Width of Petal (cm)")
ax.set_ylabel("Frequency")

# title
plt.title("Petal Widths")

#plt.show()
plt.savefig("histogram_of_petal_widths.png")

# ************************************* Investigating Relationships using Scatter Plots ****************************************************************

# Petal Lengths vs Petal Widths (Scatter Plot)
fig, ax = plt.subplots()

#Scatter plot
ax.scatter(setosa_petal_width,setosa_petal_length, marker="o",label="Setosa")
ax.scatter(versicolor_petal_width,versicolor_petal_length,marker="d", label="Versicolor")
ax.scatter(virginica_petal_width,virginica_petal_length,marker="v", label="Virginica")

# Labels
ax.set_xlabel("Petal Widths")
ax.set_ylabel("Petal Lengths")

# legend
ax.legend()

#plt.title("Relationship between Petal Lengths & Widths")
#plt.show()
plt.savefig("scatter_plot_petal_lengths_vs_petal_widths.png")

# Sepal Lengths vs Sepal Widths (Scatter Plot)
fig, ax = plt.subplots()

#Scatter plot
ax.scatter(setosa_sepal_width,setosa_sepal_length, marker="o",label="Setosa")
ax.scatter(versicolor_sepal_width,versicolor_sepal_length,marker="d", label="Versicolor")
ax.scatter(virginica_sepal_width,virginica_sepal_length,marker="v", label="Virginica")

# Labels
ax.set_xlabel("Sepal Widths")
ax.set_ylabel("Sepal Lengths")
ax.legend()

plt.title("Relationship between Sepal Lengths & Widths")
#plt.show()
plt.savefig("scatter_plot_sepal_lengths_vs_sepal_widths.png")


# ************************************************* Analyze Relationships Using Fitted Lines *************************************************************

# From https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html, 
# "Numpy Polyfit fits a polynomial p(x) = p[0] * x**deg + ... + p[deg] of degree deg to points (x, y). 
# It returns a vector of coefficients p that minimises the squared error in the order deg, deg-1, … 0"
# This is what i want, a line which when drawn through the data, on average the observations both above and below will equate

# Petal Lengths vs Petal Widths (Regression line)

# To the polyfit function, pass both arrays, and request one degree of the fitted line, (one degree as it is line not a U-shaped or n-shaped curve).
# Polyfit returns the slope and the intercept
setosa_petal_coefficients = np.polyfit(setosa_petal_width, setosa_petal_length, 1) # this gives an array containing two values, the slope and the intercept
versicolor_petal_coefficients = np.polyfit(versicolor_petal_width, versicolor_petal_length, 1) # this gives an array containing two values, the slope and the intercept
virginica_petal_coefficients = np.polyfit(virginica_petal_width, virginica_petal_length, 1) # this gives an array containing two values, the slope and the intercept

# now print out the regression lines for each class of iris, again for visual purposes i'm rounding the values 
print(f"The equations for the line for each class of iris (petal width vs petal length) are:")
print(f"\nSetosa:\t\t y = {round(setosa_petal_coefficients[0],2)}x + {round(setosa_petal_coefficients[1],2)}")
print(f"Versicolor:\t y = {round(versicolor_petal_coefficients[0],2)}x + {round(versicolor_petal_coefficients[1],2)}")
print(f"Virginica:\t y = {round(virginica_petal_coefficients[0],2)}x + {round(virginica_petal_coefficients[0],2)}")

# I know the slope and intercept coefficients of our line, I can now create y-coordinates for each point along the regression line (f) by feeding in the x-axis values.
f_setosa_petal = np.polyval(setosa_petal_coefficients,setosa_petal_width)
f_versicolor_petal = np.polyval(versicolor_petal_coefficients,versicolor_petal_width)
f_virginica_petal = np.polyval(virginica_petal_coefficients,virginica_petal_width)

# show f, which is the y-coordinates of our line through our scatterplot 
print (f"\nThe y-coordinates for equation of a line for Setosa are:\n {f_setosa_petal}\n")
print (f"The y-coordinates for equation of a line for Versicolor are:\n {f_versicolor_petal}\n")
print (f"The y-coordinates for equation of a line for Virginica are:\n{f_virginica_petal}")

# show the actual y-values so I can see how they compare with f
print (f"\nThe actual y-coordinates for Setosa are:\n {setosa_petal_length}\n")
print (f"The actual y-coordinates for Versicolor are:\n {versicolor_petal_length}\n")
print (f"The actual y-coordinates for Virginica are:\n{virginica_petal_length}")

# Petal Lengths vs Petal Widths (Scatter Plot & Regression Line)

fig, ax = plt.subplots()
# Scatter plot of each class of iris, pyplot does the heavy lighting, i just supply it with the x and y values, 
# I'm choosing differ markers to simplify distinguising them, and i'm giving them a label
ax.scatter(setosa_petal_width,setosa_petal_length, marker="o",label="Setosa")
ax.scatter(versicolor_petal_width,versicolor_petal_length,marker="d", label="Versicolor")
ax.scatter(virginica_petal_width,virginica_petal_length,marker="v", label="Virginica")

# Regression line (f) of each class of iris, I'm giving each line a colour
ax.plot(setosa_petal_width, f_setosa_petal, color='blue')
ax.plot(versicolor_petal_width, f_versicolor_petal, color='orange')
ax.plot(virginica_petal_width, f_virginica_petal, color='green')

# labels for both of the axis
ax.set_xlabel("Petal Widths")
ax.set_ylabel("Petal Lengths")

# legend
ax.legend()

# title
plt.title("Relationship between Petal Lengths & Widths")
#plt.show()
plt.savefig("fitted_line_petal_lengths_vs_petal_widths.png")


# Sepal Lengths vs Sepal Widths (Regression line)

# To the polyfit function, pass both arrays, and request one degree of the fitted line, (one degree as it is line not a U-shaped or n-shaped curve).
# Polyfit returns the slope and the intercept
setosa_sepal_coefficients = np.polyfit(setosa_sepal_width, setosa_sepal_length, 1) # this gives an array containing two values, the slope and the intercept
versicolor_sepal_coefficients = np.polyfit(versicolor_sepal_width, versicolor_sepal_length, 1) # this gives an array containing two values, the slope and the intercept
virginica_sepal_coefficients = np.polyfit(virginica_sepal_width, virginica_sepal_length, 1) # this gives an array containing two values, the slope and the intercept

# now print out our regression lines for each class of iris, again for visual purposes i'm rounding the values 
print(f"The equations for the line for each class of iris (sepal width vs sepal length) are:")
print(f"\nSetosa:\t\t y = {round(setosa_sepal_coefficients[0],2)}x + {round(setosa_sepal_coefficients[1],2)}")
print(f"Versicolor:\t y = {round(versicolor_sepal_coefficients[0],2)}x + {round(versicolor_sepal_coefficients[1],2)}")
print(f"Virginica:\t y = {round(virginica_sepal_coefficients[0],2)}x + {round(virginica_sepal_coefficients[0],2)}")

# I know the slope and intercept coefficients of our line, I can now create y-coordinates for each point along the regression line (f) by feeding in the x-axis values.
f_setosa_sepal = np.polyval(setosa_sepal_coefficients,setosa_sepal_width)
f_versicolor_sepal = np.polyval(versicolor_sepal_coefficients,versicolor_sepal_width)
f_virginica_sepal = np.polyval(virginica_sepal_coefficients,virginica_sepal_width)

# show f, which is the y-coordinates of our line through our scaterplot 
print (f"\nThe y-coordinates for equation of a line for Setosa are:\n {f_setosa_sepal}\n")
print (f"The y-coordinates for equation of a line for Versicolor are:\n {f_versicolor_sepal}\n")
print (f"The y-coordinates for equation of a line for Virginica are:\n{f_virginica_sepal}")

# show the actual y-values so I can see how they compare with f
print (f"\nThe actual y-coordinates for Setosa are:\n {setosa_sepal_length}\n")
print (f"The actual y-coordinates for Versicolor are:\n {versicolor_sepal_length}\n")
print (f"The actual y-coordinates for Virginica are:\n{virginica_sepal_length}")

# Sepal Lengths vs Sepal Widths (Scatter Plot & Regression Line)
fig, ax = plt.subplots()

# Scatter plot of each class of iris, pyplot does the heavy lighting, i just supply it with the x and y values, 
# I'm choosing differ markers to simplify distinguising them, and i'm giving them a label
ax.scatter(setosa_sepal_width,setosa_sepal_length, marker="o",label="Setosa")
ax.scatter(versicolor_sepal_width,versicolor_sepal_length,marker="d", label="Versicolor")
ax.scatter(virginica_sepal_width,virginica_sepal_length,marker="v", label="Virginica")

# Regression line (f) of each class of iris, I'm giving each line a colour
ax.plot(setosa_sepal_width, f_setosa_sepal, color='blue')
ax.plot(versicolor_sepal_width, f_versicolor_sepal, color='orange')
ax.plot(virginica_sepal_width, f_virginica_sepal, color='green')

#Labels
ax.set_xlabel("Sepal Widths")
ax.set_ylabel("Sepal Lengths")

# legend
ax.legend()

# title
plt.title("Relationship between Sepal Widths & Lengths")

#plt.show()
plt.savefig("fitted_line_sepal_lengths_vs_sepal_widths.png")

# ***********************************************************************************************************************************
# ******************************************* Compare the Different Class Distributions **********************************************

# I have created a box-plots of the petal lengths for each of the three classes. A boxplot is a chart that conveys a lot of information in a visually uncluttered and simple form.

# *************** Box plots of Petal Lengths *************************************************************************************
petal_lengths=np.array([setosa_petal_length, versicolor_petal_length,virginica_petal_length])
#print(petal_lengths)
# create figure, axis
fig, ax = plt.subplots()

# create boxplot, we will use .T because we need to transpose our data to make rows columns and columns rows because thats what boxplot expect)
ax.boxplot(petal_lengths.T)

# title
ax.set_title("Classes of Iris", fontsize=16)

# axis labels
ax.set_xlabel("Classes", fontsize=12)
ax.set_ylabel("Petal Length", fontsize=12)

# set names of x-axis ticks, just puts the names on the horizontal axis 
ax.set_xticks([1,2,3],["Setosa","Versicolor","Virginica"], fontsize=10)

# add a grid so we have horizontal dotted lines making it easier to see the values on the chart, alpha is how transparent the dotted lines are
ax.grid(axis="y", linestyle="--", alpha=0.7)
plt.savefig("boxplot_of_petal_lengths.png")

# *************** Box plots of Petal Widths *************************************************************************************
petal_widths=np.array([setosa_petal_width, versicolor_petal_width,virginica_petal_width])
#print(petal_widths)
# create figure, axis
fig, ax = plt.subplots()

# create boxplot, we will use .T because we need to transpose our data to make rows columns and columns rows because thats what boxplot expect)
ax.boxplot(petal_widths.T)

# title
ax.set_title("Classes of Iris", fontsize=16)

# axis labels
ax.set_xlabel("Classes", fontsize=12)
ax.set_ylabel("Petal Length", fontsize=12)

# set names of x-axis ticks, just puts the names on the horizontal axis 
ax.set_xticks([1,2,3],["Setosa","Versicolor","Virginica"], fontsize=10)

# add a grid so we have horizontal dotted lines making it easier to see the values on the chart, alpha is how transparent the dotted lines are
ax.grid(axis="y", linestyle="--", alpha=0.7)
plt.savefig("boxplot_of_petal_widths.png")

# *************** Box plots of Sepal Lengths *************************************************************************************
sepal_lengths=np.array([setosa_sepal_length, versicolor_sepal_length,virginica_sepal_length])
#print(petal_lengths)
# create figure, axis
fig, ax = plt.subplots()

# create boxplot, we will use .T because we need to transpose our data to make rows columns and columns rows because thats what boxplot expect)
ax.boxplot(sepal_lengths.T)

# title
ax.set_title("Classes of Iris", fontsize=16)

# axis labels
ax.set_xlabel("Classes", fontsize=12)
ax.set_ylabel("Petal Length", fontsize=12)

# set names of x-axis ticks, just puts the names on the horizontal axis 
ax.set_xticks([1,2,3],["Setosa","Versicolor","Virginica"], fontsize=10)

# add a grid so we have horizontal dotted lines making it easier to see the values on the chart, alpha is how transparent the dotted lines are
ax.grid(axis="y", linestyle="--", alpha=0.7)
plt.savefig("boxplot_of_sepal_lengths.png")

# *************** Box plots of Sepal Widths *************************************************************************************
sepal_widths=np.array([setosa_sepal_width, versicolor_sepal_width,virginica_sepal_width])
#print(petal_lengths)
# create figure, axis
fig, ax = plt.subplots()

# create boxplot, we will use .T because we need to transpose our data to make rows columns and columns rows because thats what boxplot expect)
ax.boxplot(sepal_widths.T)

# title
ax.set_title("Classes of Iris", fontsize=16)

# axis labels
ax.set_xlabel("Classes", fontsize=12)
ax.set_ylabel("Petal Length", fontsize=12)

# set names of x-axis ticks, just puts the names on the horizontal axis 
ax.set_xticks([1,2,3],["Setosa","Versicolor","Virginica"], fontsize=10)

# add a grid so we have horizontal dotted lines making it easier to see the values on the chart, alpha is how transparent the dotted lines are
ax.grid(axis="y", linestyle="--", alpha=0.7)
plt.savefig("boxplot_of_sepal_widths.png")

# ************************************************************************************************************************


# ****************** Correlation Coefficients ****************************************************************************
# The correlation coefficient is a builtin function from pandas, (source: https://www.geeksforgeeks.org/python-pandas-dataframe-corr/) 
# so we want to see how each of our variables correlate with one another
# however first we must state that from our array, we only want the numeric columns, not the alphabetical class column
# we do this by passing the "numeric_only=True" parameter in the correlation function (we'll leve the default as Pearson's correlation)
print(f"\nIris - Table of Correlation Coefficients\n{iris_df.corr(numeric_only=True,)}\n")

#print(f"{max_setosa}") # we calculated the maximum value above when creating the arrays
# we start at a minimum row of 0, up to row 50, that is, max_setosa

# so define our Setosa dataframe as the first 50 rows of the iris dataframe
setosa_df = iris_df[0:max_setosa]
#print(f"{setosa_df}")
print(f"Setosa - Table of Correlation Coefficients\n{setosa_df.corr(numeric_only=True,)}")

# so define our Versicolor dataframe as the first 50-100 rows of the iris dataframe
versicolor_df = iris_df[max_setosa:max_versicolor]
#print(f"{versicolor_df}")
print(f"\nVersicolor - Table of Correlation Coefficients\n{versicolor_df.corr(numeric_only=True,)}")

# so define our Virginica dataframe as the first 100-150 rows of the iris dataframe
virginica_df = iris_df[max_versicolor:max_virginica]
#print(f"{virginica_df}")
print(f"\nVirginica - Table of Correlation Coefficients\n{virginica_df.corr(numeric_only=True,)}\n")


# let's now write it to an external text file so we can copy it into our report if needed
with open("correlation_coefficients.txt","w") as f:
     f.write(str(iris_df.corr(numeric_only=True)))

#Correlation Coefficient Summary:
#- Sepal length and sepal width have little or no correlation, at 0.1
#- But sepal length has a relatively large correlation with both petal length and petal width (circa 0.8)
#- Next, petal length has only a small correlation with sepal width (circa 0.4)
#- Finally, we see that petal width has a very large correlation with petal length (circa 0.96) 
#***************************************************************************************************************************** 

# let's now write it to an external text file so we can copy it into our report if needed
with open("correlation_coefficients.txt","w") as f:
     f.write("\n"+"\n"+"Iris"+"\n")
     f.write(str(iris_df.corr(numeric_only=True)))
     f.write("\n"+"\n"+"Setosa"+"\n")
     f.write(str(setosa_df.corr(numeric_only=True)))
     f.write("\n"+"\n"+"Versicolor"+"\n")
     f.write(str(versicolor_df.corr(numeric_only=True)))
     f.write("\n"+"\n"+"Virginica"+"\n")
     f.write(str(virginica_df.corr(numeric_only=True)))

# **************************************************************************************************************************

# *************************************** Heat Map of Correlations **********************************************************
fig, ax = plt.subplots()

# my dataframe
data = (iris_df.corr(numeric_only=True))

#these are what we want to check how they correlate
features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

im = ax.imshow(data, cmap="seismic")
# choosing the color for color map - reference
        
# some ideas for charts here https://stackoverflow.com/questions/33282368/plotting-a-2d-heatmap 
# and https://www.geeksforgeeks.org/display-the-pandas-dataframe-in-heatmap-style/ I found  colorbar
# which is a legend for how strong the correlation is
plt.colorbar(im, label="Correlation Coefficient")

# Show all ticks and label them with the respective list entries
plt.xticks(range(len(data)), data.columns) 
plt.yticks(range(len(data)), data.columns) 

# axis labels
ax.set_xlabel("Iris Features", fontsize=12)
ax.set_ylabel("Iris Features", fontsize=12)

ax.set_title("Correlation Coefficients")
fig.tight_layout()
#plt.show()
plt.savefig("heatmap_of_iris_ correlation_coefficients.png")

# Heatmap for Setosa

fig, ax = plt.subplots()
data_setosa = (setosa_df.corr(numeric_only=True))

#these are what we want to check how they correlate
features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# choosing the color for color map
im = ax.imshow(data_setosa, cmap="seismic")

# colorbar
plt.colorbar(im, label="Correlation Coefficient")

# Show all ticks and label them with the respective list entries
plt.xticks(range(len(data_setosa)), data_setosa.columns) 
plt.yticks(range(len(data_setosa)), data_setosa.columns) 

# axis labels
ax.set_xlabel("Setosa Features", fontsize=12)
ax.set_ylabel("Setosa Features", fontsize=12)

ax.set_title("Correlation Coefficients for Setosa")
fig.tight_layout()
#plt.show()
plt.savefig("heatmap_of_setosa_correlation_coefficients.png")

# Heatmap for Versicolor
fig, ax = plt.subplots()
data_versicolor = (versicolor_df.corr(numeric_only=True))

#these are what we want to check how they correlate
features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# choosing the color for color map
im = ax.imshow(data_versicolor, cmap="seismic")

# colorbar
plt.colorbar(im, label="Correlation Coefficient")

# Show all ticks and label them with the respective list entries
plt.xticks(range(len(data_versicolor)), data_versicolor.columns) 
plt.yticks(range(len(data_versicolor)), data_versicolor.columns) 

# axis labels
ax.set_xlabel("Versicolor Features", fontsize=12)
ax.set_ylabel("Versicolor Features", fontsize=12)

ax.set_title("Correlation Coefficients for Versicolor")
fig.tight_layout()
#plt.show()
plt.savefig("heatmap_of_versicolor_correlation_coefficients.png")

# Heatmap for Virginica
fig, ax = plt.subplots()
data_virginica = (virginica_df.corr(numeric_only=True))

#these are what we want to check how they correlate
features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# choosing the color for color map
im = ax.imshow(data_virginica, cmap="seismic")

# colorbar
plt.colorbar(im, label="Correlation Coefficient")

# Show all ticks and label them with the respective list entries
plt.xticks(range(len(data_virginica)), data_virginica.columns) 
plt.yticks(range(len(data_virginica)), data_virginica.columns) 

# axis labels
ax.set_xlabel("Virginica Features", fontsize=12)
ax.set_ylabel("Virginica Features", fontsize=12)

ax.set_title("Correlation Coefficients for Virginica")
fig.tight_layout()
#plt.show()
plt.savefig("heatmap_of_virginica_correlation_coefficients.png")




#****************************************************************************************************************************
#******************************** R-Squared ********************************************************************************
#****************************************************************************************************************************

# Using the linear regression model approach as seen here https://stackoverflow.com/questions/28753502/scipy-stats-linregress-get-p-value-of-intercept
# and https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html 
# must be in the form of linregress(x variable, y=variable)

# ************************ Petal Width regressed against Petal Length **************************************************************
fig, ax = plt.subplots()

#Scatter plot
ax.scatter(setosa_petal_width,setosa_petal_length, marker="o",label="Setosa")
ax.scatter(versicolor_petal_width,versicolor_petal_length,marker="d", label="Versicolor")
ax.scatter(virginica_petal_width,virginica_petal_length,marker="v", label="Virginica")

#plt.plot(setosa_petal_width, setosa_petal_length, 'o', label='original data')
#plt.plot(versicolor_petal_width, versicolor_petal_length, 'o', label='original data')
#plt.plot(virginica_petal_width, virginica_petal_length, 'o', label='original data')

fit_setosa_petal = sp.stats.linregress(setosa_petal_width, setosa_petal_length)
#slope, intercept, r_value, p_value, std_err = stats.linregress(setosa_petal_width, setosa_petal_length)
print(f"Setosa Petal Length = b*Setosa Petal Width\nSlope is:\t{round(fit_setosa_petal.slope,2)}\nIntercept is:\t{round(fit_setosa_petal.intercept,2)}\nR-Squared is:\t{round(fit_setosa_petal.rvalue**2,2)}\nP-value is:\t{round(fit_setosa_petal.pvalue,2)}\nStd Error is:\t{round(fit_setosa_petal.stderr,2)}")
plt.plot(setosa_petal_width, fit_setosa_petal.intercept + fit_setosa_petal.slope*setosa_petal_width, 'b', label='Setosa R-squared')
      
fit_versicolor_petal = sp.stats.linregress(versicolor_petal_width, versicolor_petal_length)
print(f"\nVersicolor Petal Length = b*Versicolor Petal Width\nSlope is:\t{round(fit_versicolor_petal.slope,2)}\nIntercept is:\t{round(fit_versicolor_petal.intercept,2)}\nR-Squared is:\t{round(fit_versicolor_petal.rvalue**2,2)}\nP-value is:\t{round(fit_versicolor_petal.pvalue,2)}\nStd Error is:\t{round(fit_versicolor_petal.stderr,2)}")
plt.plot(versicolor_petal_width, fit_versicolor_petal.intercept + fit_versicolor_petal.slope*versicolor_petal_width, 'y', label='Versicolor R-squared')

fit_virginica_petal = sp.stats.linregress(virginica_petal_width, virginica_petal_length)
print(f"\nVirginica Petal Length = b*Virginica Petal Width\nSlope is:\t{round(fit_virginica_petal.slope,2)}\nIntercept is:\t{round(fit_virginica_petal.intercept,2)}\nR-Squared is:\t{round(fit_virginica_petal.rvalue**2,2)}\nP-value is:\t{round(fit_virginica_petal.pvalue,2)}\nStd Error is:\t{round(fit_virginica_petal.stderr,2)}")
plt.plot(virginica_petal_width, fit_virginica_petal.intercept + fit_virginica_petal.slope*virginica_petal_width, 'g', label='Virginica R-squared')

#Labels
ax.set_xlabel("Sepal Widths")
ax.set_ylabel("Sepal Lengths")

# title
plt.title("Relationship between Petal Widths & Lengths")

# legend
plt.legend()

# save the graph
plt.savefig("Fitted_Line_on_petal_length_vs_petal_width.png")

# ***************************************************************************************************************************************************************
# ************************************** Double Chekc my R-Squared Calculation *********************************************************************************
# Just to double-check our results for R-squared
ss_res_setosa = np.sum((setosa_petal_length - f_setosa_petal)**2)
ss_res_versicolor = np.sum((versicolor_petal_length - f_versicolor_petal)**2)
ss_res_virginica = np.sum((virginica_petal_length - f_virginica_petal)**2)

# Show
# print(f"{ss_res_setosa}")
# print(f"{ss_res_versicolor}")
# print(f"{ss_res_virginica}\n")

# Mean of petal_length
setosa_petal_length_bar = np.mean(setosa_petal_length)
versicolor_petal_length_bar = np.mean(versicolor_petal_length)
virginica_petal_length_bar = np.mean(virginica_petal_length)

# show
# print(f"{setosa_petal_length_bar}")
# print(f"{versicolor_petal_length_bar}")
# print(f"{virginica_petal_length_bar}\n")

# Total sum of squares
ss_tot_setosa = np.sum((setosa_petal_length - setosa_petal_length_bar)**2)
ss_tot_versicolor = np.sum((versicolor_petal_length - versicolor_petal_length_bar)**2)
ss_tot_virginica = np.sum((virginica_petal_length - virginica_petal_length_bar)**2)

#show
# print(f"{ss_tot_setosa}")
# print(f"{ss_tot_versicolor}")
# print(f"{ss_tot_virginica}\n")

# ratio
# print(f"ss_re/ss_tot = {ss_res_setosa/ss_tot_setosa}")
# print(f"ss_re/ss_tot = {ss_res_versicolor/ss_tot_versicolor}")
# print(f"ss_re/ss_tot = {ss_res_virginica/ss_tot_virginica}\n")

# R^2 value
print(f"\nMy R-Squared Values for Petal Widths regressed against Pelal Lengths")
print(f"Setosa = {1.0 - (ss_res_setosa / ss_tot_setosa)}")
print(f"Versicolor = {1.0 - (ss_res_versicolor / ss_tot_versicolor)}")
print(f"Virginica = {1.0 - (ss_res_virginica / ss_tot_virginica)}")

# *****************************************************************************************************************************************************************

# ************************ Sepal Width regressed against Sepal Length **************************************************************
fig, ax = plt.subplots()

#Scatter plot
ax.scatter(setosa_sepal_width,setosa_sepal_length, marker="o",label="Setosa")
ax.scatter(versicolor_sepal_width,versicolor_sepal_length,marker="d", label="Versicolor")
ax.scatter(virginica_sepal_width,virginica_sepal_length,marker="v", label="Virginica")

#plt.plot(setosa_sepal_width, setosa_sepal_length, 'o', label='original data')
#plt.plot(versicolor_sepal_width, versicolor_sepal_length, 'd', label='original data')
#plt.plot(virginica_sepal_width, virginica_sepal_length, 'o', label='original data')

fit_setosa_sepal = sp.stats.linregress(setosa_sepal_width, setosa_sepal_length)
print(f"\nSetosa Sepal Length = b*Setosa Sepal Width\nSlope is:\t{round(fit_setosa_sepal.slope,2)}\nIntercept is:\t{round(fit_setosa_sepal.intercept,2)}\nR-Squared is:\t{round(fit_setosa_sepal.rvalue**2,2)}\nP-value is:\t{round(fit_setosa_sepal.pvalue,2)}\nStd Error is:\t{round(fit_setosa_sepal.stderr,2)}")
plt.plot(setosa_sepal_width, fit_setosa_sepal.intercept + fit_setosa_sepal.slope*setosa_sepal_width, 'b', label='Setosa R-squared')

fit_versicolor_sepal = sp.stats.linregress(versicolor_sepal_width, versicolor_sepal_length)
print(f"\nVersicolor Sepal Length = b*Versicolor Sepal Width\nSlope is:\t{round(fit_versicolor_sepal.slope,2)}\nIntercept is:\t{round(fit_versicolor_sepal.intercept,2)}\nR-Squared is:\t{round(fit_versicolor_sepal.rvalue**2,2)}\nP-value is:\t{round(fit_versicolor_sepal.pvalue,2)}\nStd Error is:\t{round(fit_versicolor_sepal.stderr,2)}")
plt.plot(versicolor_sepal_width, fit_versicolor_sepal.intercept + fit_versicolor_sepal.slope*versicolor_sepal_width, 'y', label='Versicolor R-squared')

fit_virginica_sepal = sp.stats.linregress(virginica_sepal_width, virginica_sepal_length)
print(f"\nVirginica Sepal Length = b*Virginica Sepal Width\nSlope is:\t{round(fit_virginica_sepal.slope,2)}\nIntercept is:\t{round(fit_virginica_sepal.intercept,2)}\nR-Squared is:\t{round(fit_virginica_sepal.rvalue**2,2)}\nP-value is:\t{round(fit_virginica_sepal.pvalue,2)}\nStd Error is:\t{round(fit_virginica_sepal.stderr,2)}")
plt.plot(virginica_sepal_width, fit_virginica_sepal.intercept + fit_virginica_sepal.slope*virginica_sepal_width, 'g', label='Verginica R-squared')

#Labels
ax.set_xlabel("Sepal Widths")
ax.set_ylabel("Sepal Lengths")
ax.legend()

plt.title("Relationship between Sepal Widths & Lengths")
plt.legend()
plt.savefig("Fitted_Line_on_sepal_length_vs_sepal_width.png")

# *************************************************************************************************************************************************

# *****************************************Double checking my R-Squared Calculation ********************************************************************

# Just to double-check our results for R-squared
ss_res_setosa_sepal = np.sum((setosa_sepal_length - f_setosa_sepal)**2)
ss_res_versicolor_sepal = np.sum((versicolor_sepal_length - f_versicolor_sepal)**2)
ss_res_virginica_sepal = np.sum((virginica_sepal_length - f_virginica_sepal)**2)

# Show
# print(f"{ss_res_setosa_sepal}")
# print(f"{ss_res_versicolor_sepal}")
# print(f"{ss_res_virginica_sepal}\n")

# Mean of sepal_length
setosa_sepal_length_bar = np.mean(setosa_sepal_length)
versicolor_sepal_length_bar = np.mean(versicolor_sepal_length)
virginica_sepal_length_bar = np.mean(virginica_sepal_length)

# show the mean sepal length
# print(f"{setosa_petal_length_bar}")
# print(f"{versicolor_petal_length_bar}")
# print(f"{virginica_petal_length_bar}\n")

# Total sum of squares
ss_tot_setos_sepal = np.sum((setosa_sepal_length - setosa_sepal_length_bar)**2)
ss_tot_versicolor_sepal = np.sum((versicolor_sepal_length - versicolor_sepal_length_bar)**2)
ss_tot_virginica_sepal = np.sum((virginica_sepal_length - virginica_sepal_length_bar)**2)

#show
# print(f"{ss_tot_setos_sepal}")
# print(f"{ss_tot_versicolor_sepal}")
# print(f"{ss_tot_virginica_sepal}\n")

# ratio 
# print(f"ss_re/ss_tot = {ss_res_setosa_sepal/ss_tot_setos_sepal}")
# print(f"ss_re/ss_tot = {ss_res_versicolor_sepal/ss_tot_versicolor_sepal}")
# print(f"ss_re/ss_tot = {ss_res_virginica_sepal/ss_tot_virginica_sepal}\n")

# R^2 value
print(f"\nMy R-Squared Values for Sepal Widths regressed against Sepal Lengths")
print(f"Setosa = {1.0 - (ss_res_setosa_sepal / ss_tot_setos_sepal)}")
print(f"Versicolor = {1.0 - (ss_res_versicolor_sepal / ss_tot_versicolor_sepal)}")
print(f"Virginica = {1.0 - (ss_res_virginica_sepal / ss_tot_virginica_sepal)}")

# ******************************************************************************************************************************************************




# *************************************** Add a Pairplot **********************************************************************************************
# To summarise many of these graphs into one, I will add a pairplot.
# *****************************************************************************************************************************************************
# the following are based on examples from https://seaborn.pydata.org/generated/seaborn.pairplot.html 
# and https://seaborn.pydata.org/tutorial/introduction.html


g=sns.pairplot(iris_df,  x_vars=["sepal_length","sepal_width","petal_length","petal_width"],
    y_vars=["sepal_length","sepal_width","petal_length","petal_width"], hue="class",diag_kind="hist", markers=["o", "s", "D"], corner=False, height=1.5)

# legend
g.add_legend(frameon=True)

# save the file
plt.savefig("pairplot.png")
