# This programs reads in the data from the iris dataset
# it outputs a summary of each variable
# outputs a scatter plot of each pair of variables
# performs a linear regression to confirm the relationship between any two variables

# i have the iris data in a data subfolder

# author: gerry callaghan

# I will import matplotlib so i can use this library for statisics, plotting etc
import matplotlib.pyplot as plt
# I will import numpy so i can perform various mathematical operations with my arrays 
import numpy as np
# I will import pandas because it is great for with dataframes
import pandas as pd
# sklearn is a popular machine learning library that provides tools for data preprocessing, model selection, and evaluation.
from sklearn.linear_model import LinearRegression as lr

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
col_names= ("sepal_length", "sepal_width", "petal_length", "petal_width", "class")
# create a dataframe (df) and set it equal to our imported data from the this file iris.csv
# the data is column separated, the csv file currently has no headers, sp let's add the column names as we import the data
iris_df = pd.read_csv(csv_filename, sep=',',  header= None, names=col_names)

         

'''
print(f"{iris_df.shape}")
#'''
# the print function has showed us there are 150 rows of data x 5 columns, 

# let's eyeball the data so we can see it has imported okay
#'''
#print(f"{iris_df}") 
#'''
# we can see the first four columns are numeric data comprising the sepal/petal length/width and then we have a column of alphabetical data in the iris class
# More importantly, we can see that everything has been sorted by the class column, so no need for us to sort again

# Let's use drop na to get rid of the values in the series that have no value
    # this actually returns a numpy.ndarray
iris_df.dropna(inplace=True)
#let's show the data again to see if there was any change
#'''
#print(f"{iris_df}") 
#'''
# everything looks fine for us to work with the data now

#now to confirm the number of different classes of iris, we use the numpy unique function on the column titled "class"
'''
print(f"{iris_df["class"].nunique()}")
'''
# so there are 3 types, which matches with the attribute information in the data file 
# which says there are Iris Setosa, Iris Versicolour, and Iris Virginica
# we can be more confident that there were no typos in the class names

#Just want to doublecheck the number of instances of class
print(f"{iris_df["class"].size}") # this tells there are 150 class observations

#'''
# According to the file "iris.names" which was in the zip file, we should get the following
# 5. Number of Instances: 150 (50 in each of three classes)

# Let's confirm by using a quick while command (I could perhaps just create a function and call the function for each iris class )
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
 
print (f"\nClass of Iris\tObervations \nSetosa:\t\t{count_setosa}\nVersicolor:\t{count_versicolor} \nVirginica:\t{count_virginica}\n")
#'''
# so now we know there are 150 observations, 5 columns of which 4 are numeric and one alphabetical, 
# 3 classes and definitely 50 instances of each class, which confirms the iris.names.data file

#summary=df.describe()
#print(f"{summary}")

# A function for creating arrays for each class of iris
def creating_array(DataFrame,min,max):
    row_number = min # starting row of the dataframe for the relevant iris class
    output_array = [] # initilise the final array as being blank
    
    while row_number < max: # while the row number of the dataframe is less than the last row of data for that respective iris class (0-50,50-100,100-150)
         current_row_of_data = iris_df["sepal_length"][row_number],iris_df["sepal_width"][row_number],iris_df["petal_length"][row_number],iris_df["petal_width"][row_number]
         row_number = row_number + 1
         output_array = output_array + [current_row_of_data]
    #current_array = df["sepal_length"][row_number],df["sepal_width"][row_number],df["petal_length"][row_number],df["petal_width"][row_number]
    #output_array = output_array + [current_array] 
    return output_array

# so now I'm going to call the above function to create an array for each class, 
# This function creats_array to incrementally append each line of the spreadsheet to the existing array
# I'm going to do what i would do in Visual Basic and do it long hand because the append function won't work for me :-( 

#setosa = [] # create the blank setosa array
#versicolor = [] # create the blank versicolor array
#virginica =[] # create the blank virginica array
  
setosa = np.array(creating_array(iris_df, min=0,max=50)) # so to the function, we send the dataframe, the starting point and the last row for setosa which is 50
max = count_setosa + count_versicolor # so now we know versicolor starts after setosa, so the last row must be where count setosa + count versicolor combined, ie 100
versicolor = np.array(creating_array(iris_df, min=50,max = 100)) # so to the function, we send the dataframe, the starting point and the end point for versicolor
max = max + count_virginica # so now we know virginica starts after versicolor, so the last row must be the previous two combined + count versicolor combined, ie 150
virginica = np.array(creating_array(iris_df, min=100,max =150)) # so to the function, we send the dataframe, the starting point and the end point for virginica


# we can compare our each of our arrays against that of source spreadsheet
#print(f"The array for Setosa is:\n{setosa}\n\nThe array for Versicolor is:\n{versicolor}\n\nThe array for Virginica is:\n{virginica}")
# the rows all look good for each class

# Let's now create an array for iris, which in turn comprises the three arrays for setosa, versicolor and virginica
# iris = [setosa,versicolor,virginica]
# print(f"This is what an overall array comprising of 150 observations looks like: {iris}")
# okay, this matches the source data

# We want to output a summary of each variable 
# where I'm assuming each variable means feature "sepal_length", "sepal_width", "petal_length", and "petal_width"


#*************************************************************************************************************************************
# Sepal Lengths
#*************************************************************************************************************************************
setosa_sepal_length=np.array([setosa[0][0],setosa[1][0],setosa[2][0],setosa[3][0],setosa[4][0],setosa[5][0],setosa[6][0],setosa[7][0],setosa[8][0],setosa[9][0],setosa[10][0],setosa[11][0],setosa[12][0],setosa[13][0],setosa[14][0],setosa[15][0],setosa[16][0],setosa[17][0],setosa[18][0],setosa[19][0],setosa[20][0],setosa[21][0],setosa[22][0],setosa[23][0],setosa[24][0],setosa[25][0],setosa[26][0],setosa[27][0],setosa[28][0],setosa[29][0],setosa[30][0],setosa[31][0],setosa[32][0],setosa[33][0],setosa[34][0],setosa[35][0],setosa[36][0],setosa[37][0],setosa[38][0],setosa[39][0],setosa[40][0],setosa[41][0],setosa[42][0],setosa[43][0],setosa[44][0],setosa[45][0],setosa[46][0],setosa[47][0],setosa[48][0],setosa[49][0]])
print(f"Setosa Sepal Lengths are:\n {setosa_sepal_length}\n")
versicolor_sepal_length=np.array([versicolor[0][0],versicolor[1][0],versicolor[2][0],versicolor[3][0],versicolor[4][0],versicolor[5][0],versicolor[6][0],versicolor[7][0],versicolor[8][0],versicolor[9][0],versicolor[10][0],versicolor[11][0],versicolor[12][0],versicolor[13][0],versicolor[14][0],versicolor[15][0],versicolor[16][0],versicolor[17][0],versicolor[18][0],versicolor[19][0],versicolor[20][0],versicolor[21][0],versicolor[22][0],versicolor[23][0],versicolor[24][0],versicolor[25][0],versicolor[26][0],versicolor[27][0],versicolor[28][0],versicolor[29][0],versicolor[30][0],versicolor[31][0],versicolor[32][0],versicolor[33][0],versicolor[34][0],versicolor[35][0],versicolor[36][0],versicolor[37][0],versicolor[38][0],versicolor[39][0],versicolor[40][0],versicolor[41][0],versicolor[42][0],versicolor[43][0],versicolor[44][0],versicolor[45][0],versicolor[46][0],versicolor[47][0],versicolor[48][0],versicolor[49][0]])
print(f"Versicolor Sepal Lengths are:\n {versicolor_sepal_length}\n")
virginica_sepal_length=np.array([virginica[0][0],virginica[1][0],virginica[2][0],virginica[3][0],virginica[4][0],virginica[5][0],virginica[6][0],virginica[7][0],virginica[8][0],virginica[9][0],virginica[10][0],virginica[11][0],virginica[12][0],virginica[13][0],virginica[14][0],virginica[15][0],virginica[16][0],virginica[17][0],virginica[18][0],virginica[19][0],virginica[20][0],virginica[21][0],virginica[22][0],virginica[23][0],virginica[24][0],virginica[25][0],virginica[26][0],virginica[27][0],virginica[28][0],virginica[29][0],virginica[30][0],virginica[31][0],virginica[32][0],virginica[33][0],virginica[34][0],virginica[35][0],virginica[36][0],virginica[37][0],virginica[38][0],virginica[39][0],virginica[40][0],virginica[41][0],virginica[42][0],virginica[43][0],virginica[44][0],virginica[45][0],virginica[46][0],virginica[47][0],virginica[48][0],virginica[49][0]])
print(f"Virginica Sepal Lengths are:\n {virginica_sepal_length}")
overall_sepal_length= iris_df["sepal_length"]

# Descriptive Statistics
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

# from w3 schools Create a new file if it does not exist: f = open("myfile.txt", "w")
# so let's call our file sepal_lengths.txt
# this now prints the above information to a text file called sepal_lengths.txt in the same directory
# i have to convert every number to a string otherwise it won't write to the file
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

fig, ax = plt.subplots()
# I'm going to separate each class of iris on the historgram because it can help determine how they differ
ax.hist(setosa_sepal_length, bins=10, edgecolor="black", label="Setosa", alpha=0.5)
ax.hist(versicolor_sepal_length, bins=10, edgecolor="black", label="Versicolor", alpha=0.5)
ax.hist(virginica_sepal_length, bins=10, edgecolor="black", label="Virginica", alpha=0.5)
ax.legend()
ax.set_xlabel("Length of Sepal (cm)")
ax.set_ylabel("Frequency")
plt.title("Sepal Lengths")
#plt.show()
#plt.savefig("sepal_lengths.png")


#*************************************************************************************************************************************
# Sepal Widths
#*************************************************************************************************************************************

setosa_sepal_width=np.array([setosa[0][1],setosa[1][1],setosa[2][1],setosa[3][1],setosa[4][1],setosa[5][1],setosa[6][1],setosa[7][1],setosa[8][1],setosa[9][1],setosa[10][1],setosa[11][1],setosa[12][1],setosa[13][1],setosa[14][1],setosa[15][1],setosa[16][1],setosa[17][1],setosa[18][1],setosa[19][1],setosa[20][1],setosa[21][1],setosa[22][1],setosa[23][1],setosa[24][1],setosa[25][1],setosa[26][1],setosa[27][1],setosa[28][1],setosa[29][1],setosa[30][1],setosa[31][1],setosa[32][1],setosa[33][1],setosa[34][1],setosa[35][1],setosa[36][1],setosa[37][1],setosa[38][1],setosa[39][1],setosa[40][1],setosa[41][1],setosa[42][1],setosa[43][1],setosa[44][1],setosa[45][1],setosa[46][1],setosa[47][1],setosa[48][1],setosa[49][1]])
print(f"\nSetosa Sepal widths are:\n {setosa_sepal_width}\n")
versicolor_sepal_width=np.array([versicolor[0][1],versicolor[1][1],versicolor[2][1],versicolor[3][1],versicolor[4][1],versicolor[5][1],versicolor[6][1],versicolor[7][1],versicolor[8][1],versicolor[9][1],versicolor[10][1],versicolor[11][1],versicolor[12][1],versicolor[13][1],versicolor[14][1],versicolor[15][1],versicolor[16][1],versicolor[17][1],versicolor[18][1],versicolor[19][1],versicolor[20][1],versicolor[21][1],versicolor[22][1],versicolor[23][1],versicolor[24][1],versicolor[25][1],versicolor[26][1],versicolor[27][1],versicolor[28][1],versicolor[29][1],versicolor[30][1],versicolor[31][1],versicolor[32][1],versicolor[33][1],versicolor[34][1],versicolor[35][1],versicolor[36][1],versicolor[37][1],versicolor[38][1],versicolor[39][1],versicolor[40][1],versicolor[41][1],versicolor[42][1],versicolor[43][1],versicolor[44][1],versicolor[45][1],versicolor[46][1],versicolor[47][1],versicolor[48][1],versicolor[49][1]])
print(f"Versicolor Sepal widths are:\n {versicolor_sepal_width}\n")
virginica_sepal_width=np.array([virginica[0][1],virginica[1][1],virginica[2][1],virginica[3][1],virginica[4][1],virginica[5][1],virginica[6][1],virginica[7][1],virginica[8][1],virginica[9][1],virginica[10][1],virginica[11][1],virginica[12][1],virginica[13][1],virginica[14][1],virginica[15][1],virginica[16][1],virginica[17][1],virginica[18][1],virginica[19][1],virginica[20][1],virginica[21][1],virginica[22][1],virginica[23][1],virginica[24][1],virginica[25][1],virginica[26][1],virginica[27][1],virginica[28][1],virginica[29][1],virginica[30][1],virginica[31][1],virginica[32][1],virginica[33][1],virginica[34][1],virginica[35][1],virginica[36][1],virginica[37][1],virginica[38][1],virginica[39][1],virginica[40][1],virginica[41][1],virginica[42][1],virginica[43][1],virginica[44][1],virginica[45][1],virginica[46][1],virginica[47][1],virginica[48][1],virginica[49][1]])
print(f"Virginica Sepal widths are:\n {virginica_sepal_width}")
overall_sepal_width= iris_df["sepal_width"]

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

fig, ax = plt.subplots()
# I'm going to separate each class of iris on the historgram because it can help determine how they differ
ax.hist(setosa_sepal_width, bins=10, edgecolor="black", label="Setosa", alpha=0.5)
ax.hist(versicolor_sepal_width, bins=10, edgecolor="black", label="Versicolor", alpha=0.5)
ax.hist(virginica_sepal_width, bins=10, edgecolor="black", label="Virginica", alpha=0.5)
ax.legend()
ax.set_xlabel("Width of Sepal (cm)")
ax.set_ylabel("Frequency")
plt.title("Sepal Widths")
#plt.show()
#plt.savefig("sepal_widths.png")

#*************************************************************************************************************************************
# Petal Lengths
#*************************************************************************************************************************************
setosa_petal_length=np.array([setosa[0][2],setosa[1][2],setosa[2][2],setosa[3][2],setosa[4][2],setosa[5][2],setosa[6][2],setosa[7][2],setosa[8][2],setosa[9][2],setosa[10][2],setosa[11][2],setosa[12][2],setosa[13][2],setosa[14][2],setosa[15][2],setosa[16][2],setosa[17][2],setosa[18][2],setosa[19][2],setosa[20][2],setosa[21][2],setosa[22][2],setosa[23][2],setosa[24][2],setosa[25][2],setosa[26][2],setosa[27][2],setosa[28][2],setosa[29][2],setosa[30][2],setosa[31][2],setosa[32][2],setosa[33][2],setosa[34][2],setosa[35][2],setosa[36][2],setosa[37][2],setosa[38][2],setosa[39][2],setosa[40][2],setosa[41][2],setosa[42][2],setosa[43][2],setosa[44][2],setosa[45][2],setosa[46][2],setosa[47][2],setosa[48][2],setosa[49][2]])
print(f"\nSetosa Petal Lengths are:\n {setosa_petal_length}")
versicolor_petal_length=np.array([versicolor[0][2],versicolor[1][2],versicolor[2][2],versicolor[3][2],versicolor[4][2],versicolor[5][2],versicolor[6][2],versicolor[7][2],versicolor[8][2],versicolor[9][2],versicolor[10][2],versicolor[11][2],versicolor[12][2],versicolor[13][2],versicolor[14][2],versicolor[15][2],versicolor[16][2],versicolor[17][2],versicolor[18][2],versicolor[19][2],versicolor[20][2],versicolor[21][2],versicolor[22][2],versicolor[23][2],versicolor[24][2],versicolor[25][2],versicolor[26][2],versicolor[27][2],versicolor[28][2],versicolor[29][2],versicolor[30][2],versicolor[31][2],versicolor[32][2],versicolor[33][2],versicolor[34][2],versicolor[35][2],versicolor[36][2],versicolor[37][2],versicolor[38][2],versicolor[39][2],versicolor[40][2],versicolor[41][2],versicolor[42][2],versicolor[43][2],versicolor[44][2],versicolor[45][2],versicolor[46][2],versicolor[47][2],versicolor[48][2],versicolor[49][2]])
print(f"Versicolor Petal Lengths are:\n {versicolor_petal_length}\n")
virginica_petal_length=np.array([virginica[0][2],virginica[1][2],virginica[2][2],virginica[3][2],virginica[4][2],virginica[5][2],virginica[6][2],virginica[7][2],virginica[8][2],virginica[9][2],virginica[10][2],virginica[11][2],virginica[12][2],virginica[13][2],virginica[14][2],virginica[15][2],virginica[16][2],virginica[17][2],virginica[18][2],virginica[19][2],virginica[20][2],virginica[21][2],virginica[22][2],virginica[23][2],virginica[24][2],virginica[25][2],virginica[26][2],virginica[27][2],virginica[28][2],virginica[29][2],virginica[30][2],virginica[31][2],virginica[32][2],virginica[33][2],virginica[34][2],virginica[35][2],virginica[36][2],virginica[37][2],virginica[38][2],virginica[39][2],virginica[40][2],virginica[41][2],virginica[42][2],virginica[43][2],virginica[44][2],virginica[45][2],virginica[46][2],virginica[47][2],virginica[48][2],virginica[49][2]])
print(f"Virginica Petal Lengths are:\n {virginica_petal_length}")
overall_petal_length=iris_df["petal_length"]

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

fig, ax = plt.subplots()
# I'm going to separate each class of iris on the historgram because it can help determine how they differ
ax.hist(setosa_petal_length, bins=10, edgecolor="black", label="Setosa", alpha=0.5)
ax.hist(versicolor_petal_length, bins=10, edgecolor="black", label="Versicolor", alpha=0.5)
ax.hist(virginica_petal_length, bins=10, edgecolor="black", label="Virginica", alpha=0.5)
ax.legend()
ax.set_xlabel("Length of Petal (cm)")
ax.set_ylabel("Frequency")
plt.title("Petal Lengths")
#plt.show()
#plt.savefig("petal_lengths.png")


#*************************************************************************************************************************************
# Petal Widths
#*************************************************************************************************************************************

setosa_petal_width=np.array([setosa[0][3],setosa[1][3],setosa[2][3],setosa[3][3],setosa[4][3],setosa[5][3],setosa[6][3],setosa[7][3],setosa[8][3],setosa[9][3],setosa[10][3],setosa[11][3],setosa[12][3],setosa[13][3],setosa[14][3],setosa[15][3],setosa[16][3],setosa[17][3],setosa[18][3],setosa[19][3],setosa[20][3],setosa[21][3],setosa[22][3],setosa[23][3],setosa[24][3],setosa[25][3],setosa[26][3],setosa[27][3],setosa[28][3],setosa[29][3],setosa[30][3],setosa[31][3],setosa[32][3],setosa[33][3],setosa[34][3],setosa[35][3],setosa[36][3],setosa[37][3],setosa[38][3],setosa[39][3],setosa[40][3],setosa[41][3],setosa[42][3],setosa[43][3],setosa[44][3],setosa[45][3],setosa[46][3],setosa[47][3],setosa[48][3],setosa[49][3]])
print(f"\nSetosa Petal Widths are:\n {setosa_petal_width}\n")
versicolor_petal_width=np.array([versicolor[0][3],versicolor[1][3],versicolor[2][3],versicolor[3][3],versicolor[4][3],versicolor[5][3],versicolor[6][3],versicolor[7][3],versicolor[8][3],versicolor[9][3],versicolor[10][3],versicolor[11][3],versicolor[12][3],versicolor[13][3],versicolor[14][3],versicolor[15][3],versicolor[16][3],versicolor[17][3],versicolor[18][3],versicolor[19][3],versicolor[20][3],versicolor[21][3],versicolor[22][3],versicolor[23][3],versicolor[24][3],versicolor[25][3],versicolor[26][3],versicolor[27][3],versicolor[28][3],versicolor[29][3],versicolor[30][3],versicolor[31][3],versicolor[32][3],versicolor[33][3],versicolor[34][3],versicolor[35][3],versicolor[36][3],versicolor[37][3],versicolor[38][3],versicolor[39][3],versicolor[40][3],versicolor[41][3],versicolor[42][3],versicolor[43][3],versicolor[44][3],versicolor[45][3],versicolor[46][3],versicolor[47][3],versicolor[48][3],versicolor[49][3]])
print(f"Versicolor Petal Widths are:\n {versicolor_petal_width}\n")
virginica_petal_width=np.array([virginica[0][3],virginica[1][3],virginica[2][3],virginica[3][3],virginica[4][3],virginica[5][3],virginica[6][3],virginica[7][3],virginica[8][3],virginica[9][3],virginica[10][3],virginica[11][3],virginica[12][3],virginica[13][3],virginica[14][3],virginica[15][3],virginica[16][3],virginica[17][3],virginica[18][3],virginica[19][3],virginica[20][3],virginica[21][3],virginica[22][3],virginica[23][3],virginica[24][3],virginica[25][3],virginica[26][3],virginica[27][3],virginica[28][3],virginica[29][3],virginica[30][3],virginica[31][3],virginica[32][3],virginica[33][3],virginica[34][3],virginica[35][3],virginica[36][3],virginica[37][3],virginica[38][3],virginica[39][3],virginica[40][3],virginica[41][3],virginica[42][3],virginica[43][3],virginica[44][3],virginica[45][3],virginica[46][3],virginica[47][3],virginica[48][3],virginica[49][3]])
print(f"Virginica Petal Widths are:\n {virginica_petal_width}")
overall_petal_width=iris_df["petal_width"]

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
print(f"Maximum:\t{round(np.max(setosa_petal_width),2)}\t\t{round(np.max(versicolor_petal_width),2)}\t\t{round(np.max(virginica_petal_width),2)}\t\t{round(np.max(overall_petal_width),2)}")

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

fig, ax = plt.subplots()
# I'm going to separate each class of iris on the historgram because it can help determine how they differ
ax.hist(setosa_petal_width, bins=10, edgecolor="black", label="Setosa", alpha=0.5)
ax.hist(versicolor_petal_width, bins=10, edgecolor="black", label="Versicolor", alpha=0.5)
ax.hist(virginica_petal_width, bins=10, edgecolor="black", label="Virginica", alpha=0.5)
ax.legend()
ax.set_xlabel("Width of Petal (cm)")
ax.set_ylabel("Frequency")
plt.title("Petal Widths")
#plt.show()
#plt.savefig("petal_widths.png")


# ************************************************************************************************************************************************
# Petal Lengths vs Petal Widths (Scatter Plot)
# ***************************************************************************************************************************************************

fig, ax = plt.subplots()

#Scatter plot
ax.scatter(setosa_petal_width,setosa_petal_length, marker="o",label="Setosa")
ax.scatter(versicolor_petal_width,versicolor_petal_length,marker="d", label="Versicolor")
ax.scatter(virginica_petal_width,virginica_petal_length,marker="v", label="Virginica")

# Labels
ax.set_xlabel("Petal Widths")
ax.set_ylabel("Petal Lengths")
ax.legend()

#plt.title("Relationship between Petal Lengths & Widths")
#plt.show()
#plt.savefig("petal_lengths_vs_petal_widths.png")

# Use polyfit to fit a line to the data.
m1, c1 = np.polyfit(setosa_petal_width, setosa_petal_length, 1)
m2, c2 = np.polyfit(versicolor_petal_width, versicolor_petal_length, 1)
m3, c3 = np.polyfit(virginica_petal_width, virginica_petal_length, 1)

print(f"\nSetosa:\t\t y = {round(m1,2)}x + {round(c1,2)}")
print(f"Versicolor:\t y = {round(m2,2)}x + {round(c2,2)}")
print(f"Virginica:\t y = {round(m3,2)}x + {round(c3,2)}\n")

# We know what our slope and intercept coefficients are now, we can now create y-coordinates for each value of our petal widths.
y1 = setosa_petal_width* m1 + c1
y2 = versicolor_petal_width* m2 + c2
y3 = virginica_petal_width* m3 + c3

print (f"The y-coordinates for equation of a line for Setosa are:\n {y1}\n")
print (f"The y-coordinates for equation of a line for Versicolor are:\n {y2}\n")
print (f"The y-coordinates for equation of a line for Virginica are:\n{y3}")

fig, ax = plt.subplots()

#Scatter plot
ax.scatter(setosa_petal_width,setosa_petal_length, marker="o",label="Setosa")
ax.scatter(versicolor_petal_width,versicolor_petal_length,marker="d", label="Versicolor")
ax.scatter(virginica_petal_width,virginica_petal_length,marker="v", label="Virginica")
ax.plot(setosa_petal_width, y1, color='blue')
ax.plot(versicolor_petal_width, y2, color='orange')
ax.plot(virginica_petal_width, y3, color='green')

#Labels
ax.set_xlabel("Petal Widths")
ax.set_ylabel("Petal Lengths")
ax.legend()

plt.title("Relationship between Petal Widths & Lengths")
#plt.show()
plt.savefig("petal_lengths_vs_petal_widths.png")


# ************************************************************************************************************************************************
# Sepal Lengths vs Sepal Widths (Scatter Plot)
# ***************************************************************************************************************************************************

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
#plt.savefig("sepal_lengths_vs_sepal_widths.png")


# Use polyfit to fit a line to the data.
m1, c1 = np.polyfit(setosa_sepal_width, setosa_sepal_length, 1)
m2, c2 = np.polyfit(versicolor_sepal_width, versicolor_sepal_length, 1)
m3, c3 = np.polyfit(virginica_sepal_width, virginica_sepal_length, 1)

print(f"\nSetosa:\t\t y = {round(m1,2)}x + {round(c1,2)}")
print(f"Versicolor:\t y = {round(m2,2)}x + {round(c2,2)}")
print(f"Virginica:\t y = {round(m3,2)}x + {round(c3,2)}\n")

# We know what our slope and intercept coefficients are now, we can now create y-coordinates for each value of our sepal widths.
y1 = setosa_sepal_width* m1 + c1
y2 = versicolor_sepal_width* m2 + c2
y3 = virginica_sepal_width* m3 + c3

print (f"The y-coordinates for equation of a line for Setosa are:\n {y1}\n")
print (f"The y-coordinates for equation of a line for Versicolor are:\n {y2}\n")
print (f"The y-coordinates for equation of a line for Virginica are:\n{y3}")

fig, ax = plt.subplots()

#Scatter plot
ax.scatter(setosa_sepal_width,setosa_sepal_length, marker="o",label="Setosa")
ax.scatter(versicolor_sepal_width,versicolor_sepal_length,marker="d", label="Versicolor")
ax.scatter(virginica_sepal_width,virginica_sepal_length,marker="v", label="Virginica")
ax.plot(setosa_sepal_width, y1, color='blue')
ax.plot(versicolor_sepal_width, y2, color='orange')
ax.plot(virginica_sepal_width, y3, color='green')

#Labels
ax.set_xlabel("Sepal Widths")
ax.set_ylabel("Sepal Lengths")
ax.legend()

plt.title("Relationship between Sepal Widths & Lengths")
#plt.show()
plt.savefig("sepal_lengths_vs_sepal_widths.png")

# *************************************************************************************************************************************************************
# *************** Box plots of Petal Lengths *************************************************************************************
# *************************************************************************************************************************************************************

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

# *************************************************************************************************************************************************************
# *************** Box plots of Petal Widths *************************************************************************************
# *************************************************************************************************************************************************************

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

# *************************************************************************************************************************************************************
# *************** Box plots of Sepal Lengths *************************************************************************************
# *************************************************************************************************************************************************************

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

# *************************************************************************************************************************************************************
# *************** Box plots of Sepal Widths *************************************************************************************
# *************************************************************************************************************************************************************

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
# ************************************************************************************************************************

# The correlation coefficient is a builtin function from pandas, (source: https://www.geeksforgeeks.org/python-pandas-dataframe-corr/) 
# so we want to see how each of our variables correlate with one another
# however first we must state that from our array, we only want the numeric columns, not the alphabetical class column
# we do this by passing the "numeric_only=True" parameter in the correlation function (we'll leve the default as Pearson's correlation)
print(f"\n{iris_df.corr(numeric_only=True,)}\n")

# let's now write it to an external text file so we can copy it into our report if needed
with open("correlation_coefficients.txt","w") as f:
     f.write(str(iris_df.corr(numeric_only=True)))

#Correlation Coefficient Summary:
#- Sepal length and sepal width have little or no correlation, at 0.1
#- But sepal length has a relatively large correlation with both petal length and petal width (circa 0.8)
#- Next, petal length has only a small correlation with sepal width (circa 0.4)
#- Finally, we see that petal width has a very large correlation with petal length (circa 0.96) 
#*****************************************************************************************************************************
 

 #*************************************** Heat Map of Correlations **********************************************************
fig, ax = plt.subplots()
data = (iris_df.corr(numeric_only=True))

features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


im = ax.imshow(data, cmap="seismic")
# choosing the color for color map - reference
        
# okay, just messing around with some of the charts here https://stackoverflow.com/questions/33282368/plotting-a-2d-heatmap 
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
plt.savefig("heatmap_of_correlation_coefficients.png")

#****************************************************************************************************************************
#******************************** R-Squared ********************************************************************************
#****************************************************************************************************************************



model = lr()

# Train the model
model.fit(sepal_widths,sepal_lengths)

# Evaluate the model
r2_score = model.score(sepal_widths,sepal_lengths)
print(f"The R-squared value of sepal widths regressed aginst sepal lengths is: {r2_score}")

from sklearn.metrics import r2_score

print(f"{(r2_score(sepal_widths, sepal_lengths))}")
from scipy import stats

slope, intercept, r_value, p_value, std_err = stats.linregress(sepal_widths, sepal_lengths)
print("slope: %f    intercept: %f" % (slope, intercept))

# Train the model
model.fit(petal_widths,petal_lengths)

# Evaluate the model
r2_score = model.score(petal_widths,petal_lengths)
print(f"The R-squared value of petal widths regressed aginst petal lengths is: {r2_score}")





