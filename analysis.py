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
col_names= ("sepal_length", "sepal_width", "petal_length", "petal_width", "class")
# create a dataframe (df) and set it equal to our imported data from the this file iris.csv
# the data is column separated, the csv file currently has no headers, sp let's add the column names as we import the data
iris_df = pd.read_csv(csv_filename, sep=',',  header= None, names=col_names)

#'''
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
print(f"Virginica Sepal Lengths are:\n {virginica_sepal_length}\n")
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
plt.savefig("sepal_lengths.png")


#*************************************************************************************************************************************
# Sepal Widths
#*************************************************************************************************************************************

setosa_sepal_width=np.array([setosa[0][1],setosa[1][1],setosa[2][1],setosa[3][1],setosa[4][1],setosa[5][1],setosa[6][1],setosa[7][1],setosa[8][1],setosa[9][1],setosa[10][1],setosa[11][1],setosa[12][1],setosa[13][1],setosa[14][1],setosa[15][1],setosa[16][1],setosa[17][1],setosa[18][1],setosa[19][1],setosa[20][1],setosa[21][1],setosa[22][1],setosa[23][1],setosa[24][1],setosa[25][1],setosa[26][1],setosa[27][1],setosa[28][1],setosa[29][1],setosa[30][1],setosa[31][1],setosa[32][1],setosa[33][1],setosa[34][1],setosa[35][1],setosa[36][1],setosa[37][1],setosa[38][1],setosa[39][1],setosa[40][1],setosa[41][1],setosa[42][1],setosa[43][1],setosa[44][1],setosa[45][1],setosa[46][1],setosa[47][1],setosa[48][1],setosa[49][1]])
print(f"\nSetosa Sepal widths are:\n {setosa_sepal_width}\n")
versicolor_sepal_width=np.array([versicolor[0][1],versicolor[1][1],versicolor[2][1],versicolor[3][1],versicolor[4][1],versicolor[5][1],versicolor[6][1],versicolor[7][1],versicolor[8][1],versicolor[9][1],versicolor[10][1],versicolor[11][1],versicolor[12][1],versicolor[13][1],versicolor[14][1],versicolor[15][1],versicolor[16][1],versicolor[17][1],versicolor[18][1],versicolor[19][1],versicolor[20][1],versicolor[21][1],versicolor[22][1],versicolor[23][1],versicolor[24][1],versicolor[25][1],versicolor[26][1],versicolor[27][1],versicolor[28][1],versicolor[29][1],versicolor[30][1],versicolor[31][1],versicolor[32][1],versicolor[33][1],versicolor[34][1],versicolor[35][1],versicolor[36][1],versicolor[37][1],versicolor[38][1],versicolor[39][1],versicolor[40][1],versicolor[41][1],versicolor[42][1],versicolor[43][1],versicolor[44][1],versicolor[45][1],versicolor[46][1],versicolor[47][1],versicolor[48][1],versicolor[49][1]])
print(f"Versicolor Sepal widths are:\n {versicolor_sepal_width}\n")
virginica_sepal_width=np.array([virginica[0][1],virginica[1][1],virginica[2][1],virginica[3][1],virginica[4][1],virginica[5][1],virginica[6][1],virginica[7][1],virginica[8][1],virginica[9][1],virginica[10][1],virginica[11][1],virginica[12][1],virginica[13][1],virginica[14][1],virginica[15][1],virginica[16][1],virginica[17][1],virginica[18][1],virginica[19][1],virginica[20][1],virginica[21][1],virginica[22][1],virginica[23][1],virginica[24][1],virginica[25][1],virginica[26][1],virginica[27][1],virginica[28][1],virginica[29][1],virginica[30][1],virginica[31][1],virginica[32][1],virginica[33][1],virginica[34][1],virginica[35][1],virginica[36][1],virginica[37][1],virginica[38][1],virginica[39][1],virginica[40][1],virginica[41][1],virginica[42][1],virginica[43][1],virginica[44][1],virginica[45][1],virginica[46][1],virginica[47][1],virginica[48][1],virginica[49][1]])
print(f"Virginica Sepal widths are:\n {virginica_sepal_width}\n")
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
plt.savefig("sepal_widths.png")
'''
#*************************************************************************************************************************************
# Petal Lengths
#*************************************************************************************************************************************

setosa_petal_length=(iris[0][0][2],iris[0][1][2],iris[0][2][2],iris[0][3][2],iris[0][4][2],iris[0][5][2],iris[0][6][2],iris[0][7][2],iris[0][8][2],iris[0][9][2],iris[0][10][2],iris[0][11][2],iris[0][12][2],iris[0][13][2],iris[0][14][2],iris[0][15][2],iris[0][16][2],iris[0][17][2],iris[0][18][2],iris[0][19][2],iris[0][20][2],iris[0][21][2],iris[0][22][2],iris[0][23][2],iris[0][24][2],iris[0][25][2],iris[0][26][2],iris[0][27][2],iris[0][28][2],iris[0][29][2],iris[0][30][2],iris[0][31][2],iris[0][32][2],iris[0][33][2],iris[0][34][2],iris[0][35][2],iris[0][36][2],iris[0][37][2],iris[0][38][2],iris[0][39][2],iris[0][40][2],iris[0][41][2],iris[0][42][2],iris[0][43][2],iris[0][44][2],iris[0][45][2],iris[0][46][2],iris[0][47][2],iris[0][48][2],iris[0][49][2])
#print(f"Setosa Petal Lengths are: {setosa_petal_length}")
versicolor_petal_length=(iris[1][0][2],iris[1][1][2],iris[1][2][2],iris[1][3][2],iris[1][4][2],iris[1][5][2],iris[1][6][2],iris[1][7][2],iris[1][8][2],iris[1][9][2],iris[1][10][2],iris[1][11][2],iris[1][12][2],iris[1][13][2],iris[1][14][2],iris[1][15][2],iris[1][16][2],iris[1][17][2],iris[1][18][2],iris[1][19][2],iris[1][20][2],iris[1][21][2],iris[1][22][2],iris[1][23][2],iris[1][24][2],iris[1][25][2],iris[1][26][2],iris[1][27][2],iris[1][28][2],iris[1][29][2],iris[1][30][2],iris[1][31][2],iris[1][32][2],iris[1][33][2],iris[1][34][2],iris[1][35][2],iris[1][36][2],iris[1][37][2],iris[1][38][2],iris[1][39][2],iris[1][40][2],iris[1][41][2],iris[1][42][2],iris[1][43][2],iris[1][44][2],iris[1][45][2],iris[1][46][2],iris[1][47][2],iris[1][48][2],iris[1][49][2])
#print(f"Versicolor Petal Lengths are: {versicolor_petal_length}")
virginica_petal_length=(iris[2][0][2],iris[2][1][2],iris[2][2][2],iris[2][3][2],iris[2][4][2],iris[2][5][2],iris[2][6][2],iris[2][7][2],iris[2][8][2],iris[2][9][2],iris[2][10][2],iris[2][11][2],iris[2][12][2],iris[2][13][2],iris[2][14][2],iris[2][15][2],iris[2][16][2],iris[2][17][2],iris[2][18][2],iris[2][19][2],iris[2][20][2],iris[2][21][2],iris[2][22][2],iris[2][23][2],iris[2][24][2],iris[2][25][2],iris[2][26][2],iris[2][27][2],iris[2][28][2],iris[2][29][2],iris[2][30][2],iris[2][31][2],iris[2][32][2],iris[2][33][2],iris[2][34][2],iris[2][35][2],iris[2][36][2],iris[2][37][2],iris[2][38][2],iris[2][39][2],iris[2][40][2],iris[2][41][2],iris[2][42][2],iris[2][43][2],iris[2][44][2],iris[2][45][2],iris[2][46][2],iris[2][47][2],iris[2][48][2],iris[2][49][2])
#print(f"Virginica Petal Lengths are: {virginica_petal_length}")
overall_petal_length=df["petal_length"]

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
plt.title("Petal Length")
#plt.show()
plt.savefig("petal_lengths.png")


#*************************************************************************************************************************************
# Petal Widths
#*************************************************************************************************************************************

setosa_petal_width=(iris[0][0][3],iris[0][1][3],iris[0][2][3],iris[0][3][3],iris[0][4][3],iris[0][5][3],iris[0][6][3],iris[0][7][3],iris[0][8][3],iris[0][9][3],iris[0][10][3],iris[0][11][3],iris[0][12][3],iris[0][13][3],iris[0][14][3],iris[0][15][3],iris[0][16][3],iris[0][17][3],iris[0][18][3],iris[0][19][3],iris[0][20][3],iris[0][21][3],iris[0][22][3],iris[0][23][3],iris[0][24][3],iris[0][25][3],iris[0][26][3],iris[0][27][3],iris[0][28][3],iris[0][29][3],iris[0][30][3],iris[0][31][3],iris[0][32][3],iris[0][33][3],iris[0][34][3],iris[0][35][3],iris[0][36][3],iris[0][37][3],iris[0][38][3],iris[0][39][3],iris[0][40][3],iris[0][41][3],iris[0][42][3],iris[0][43][3],iris[0][44][3],iris[0][45][3],iris[0][46][3],iris[0][47][3],iris[0][48][3],iris[0][49][3])
#print(f"Setosa Petal Widths are: {setosa_petal_width}")
versicolor_petal_width=(iris[1][0][3],iris[1][1][3],iris[1][2][3],iris[1][3][3],iris[1][4][3],iris[1][5][3],iris[1][6][3],iris[1][7][3],iris[1][8][3],iris[1][9][3],iris[1][10][3],iris[1][11][3],iris[1][12][3],iris[1][13][3],iris[1][14][3],iris[1][15][3],iris[1][16][3],iris[1][17][3],iris[1][18][3],iris[1][19][3],iris[1][20][3],iris[1][21][3],iris[1][22][3],iris[1][23][3],iris[1][24][3],iris[1][25][3],iris[1][26][3],iris[1][27][3],iris[1][28][3],iris[1][29][3],iris[1][30][3],iris[1][31][3],iris[1][32][3],iris[1][33][3],iris[1][34][3],iris[1][35][3],iris[1][36][3],iris[1][37][3],iris[1][38][3],iris[1][39][3],iris[1][40][3],iris[1][41][3],iris[1][42][3],iris[1][43][3],iris[1][44][3],iris[1][45][3],iris[1][46][3],iris[1][47][3],iris[1][48][3],iris[1][49][3])
#print(f"Versicolor Petal Widths are: {versicolor_petal_width}")
virginica_petal_width=(iris[2][0][3],iris[2][1][3],iris[2][2][3],iris[2][3][3],iris[2][4][3],iris[2][5][3],iris[2][6][3],iris[2][7][3],iris[2][8][3],iris[2][9][3],iris[2][10][3],iris[2][11][3],iris[2][12][3],iris[2][13][3],iris[2][14][3],iris[2][15][3],iris[2][16][3],iris[2][17][3],iris[2][18][3],iris[2][19][3],iris[2][20][3],iris[2][21][3],iris[2][22][3],iris[2][23][3],iris[2][24][3],iris[2][25][3],iris[2][26][3],iris[2][27][3],iris[2][28][3],iris[2][29][3],iris[2][30][3],iris[2][31][3],iris[2][32][3],iris[2][33][3],iris[2][34][3],iris[2][35][3],iris[2][36][3],iris[2][37][3],iris[2][38][3],iris[2][39][3],iris[2][40][3],iris[2][41][3],iris[2][42][3],iris[2][43][3],iris[2][44][3],iris[2][45][3],iris[2][46][3],iris[2][47][3],iris[2][48][3],iris[2][49][3])
#print(f"Virginica Petal Widths are: {virginica_petal_width}")
overall_petal_width=df["petal_width"]

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
plt.title("Petal Width")
#plt.show()
plt.savefig("petal_widths.png")


# ************************************************************************************************************************************************
# Petal Lengths vs Petal Widths (Scatter Plot)
# ***************************************************************************************************************************************************

# from above, we previously calculated the petal lengths for each class of iris
setosa_petal_length=(iris[0][0][2],iris[0][1][2],iris[0][2][2],iris[0][3][2],iris[0][4][2],iris[0][5][2],iris[0][6][2],iris[0][7][2],iris[0][8][2],iris[0][9][2],iris[0][10][2],iris[0][11][2],iris[0][12][2],iris[0][13][2],iris[0][14][2],iris[0][15][2],iris[0][16][2],iris[0][17][2],iris[0][18][2],iris[0][19][2],iris[0][20][2],iris[0][21][2],iris[0][22][2],iris[0][23][2],iris[0][24][2],iris[0][25][2],iris[0][26][2],iris[0][27][2],iris[0][28][2],iris[0][29][2],iris[0][30][2],iris[0][31][2],iris[0][32][2],iris[0][33][2],iris[0][34][2],iris[0][35][2],iris[0][36][2],iris[0][37][2],iris[0][38][2],iris[0][39][2],iris[0][40][2],iris[0][41][2],iris[0][42][2],iris[0][43][2],iris[0][44][2],iris[0][45][2],iris[0][46][2],iris[0][47][2],iris[0][48][2],iris[0][49][2])
versicolor_petal_length=(iris[1][0][2],iris[1][1][2],iris[1][2][2],iris[1][3][2],iris[1][4][2],iris[1][5][2],iris[1][6][2],iris[1][7][2],iris[1][8][2],iris[1][9][2],iris[1][10][2],iris[1][11][2],iris[1][12][2],iris[1][13][2],iris[1][14][2],iris[1][15][2],iris[1][16][2],iris[1][17][2],iris[1][18][2],iris[1][19][2],iris[1][20][2],iris[1][21][2],iris[1][22][2],iris[1][23][2],iris[1][24][2],iris[1][25][2],iris[1][26][2],iris[1][27][2],iris[1][28][2],iris[1][29][2],iris[1][30][2],iris[1][31][2],iris[1][32][2],iris[1][33][2],iris[1][34][2],iris[1][35][2],iris[1][36][2],iris[1][37][2],iris[1][38][2],iris[1][39][2],iris[1][40][2],iris[1][41][2],iris[1][42][2],iris[1][43][2],iris[1][44][2],iris[1][45][2],iris[1][46][2],iris[1][47][2],iris[1][48][2],iris[1][49][2])
virginica_petal_length=(iris[2][0][2],iris[2][1][2],iris[2][2][2],iris[2][3][2],iris[2][4][2],iris[2][5][2],iris[2][6][2],iris[2][7][2],iris[2][8][2],iris[2][9][2],iris[2][10][2],iris[2][11][2],iris[2][12][2],iris[2][13][2],iris[2][14][2],iris[2][15][2],iris[2][16][2],iris[2][17][2],iris[2][18][2],iris[2][19][2],iris[2][20][2],iris[2][21][2],iris[2][22][2],iris[2][23][2],iris[2][24][2],iris[2][25][2],iris[2][26][2],iris[2][27][2],iris[2][28][2],iris[2][29][2],iris[2][30][2],iris[2][31][2],iris[2][32][2],iris[2][33][2],iris[2][34][2],iris[2][35][2],iris[2][36][2],iris[2][37][2],iris[2][38][2],iris[2][39][2],iris[2][40][2],iris[2][41][2],iris[2][42][2],iris[2][43][2],iris[2][44][2],iris[2][45][2],iris[2][46][2],iris[2][47][2],iris[2][48][2],iris[2][49][2])

# we won't be using this array, it's too disparate petal_lengths=(setosa_petal_length, versicolor_petal_length,virginica_petal_length)

# from above, we previously calculated the petal widths for each class of iris
setosa_petal_width=(iris[0][0][3],iris[0][1][3],iris[0][2][3],iris[0][3][3],iris[0][4][3],iris[0][5][3],iris[0][6][3],iris[0][7][3],iris[0][8][3],iris[0][9][3],iris[0][10][3],iris[0][11][3],iris[0][12][3],iris[0][13][3],iris[0][14][3],iris[0][15][3],iris[0][16][3],iris[0][17][3],iris[0][18][3],iris[0][19][3],iris[0][20][3],iris[0][21][3],iris[0][22][3],iris[0][23][3],iris[0][24][3],iris[0][25][3],iris[0][26][3],iris[0][27][3],iris[0][28][3],iris[0][29][3],iris[0][30][3],iris[0][31][3],iris[0][32][3],iris[0][33][3],iris[0][34][3],iris[0][35][3],iris[0][36][3],iris[0][37][3],iris[0][38][3],iris[0][39][3],iris[0][40][3],iris[0][41][3],iris[0][42][3],iris[0][43][3],iris[0][44][3],iris[0][45][3],iris[0][46][3],iris[0][47][3],iris[0][48][3],iris[0][49][3])
versicolor_petal_width=(iris[1][0][3],iris[1][1][3],iris[1][2][3],iris[1][3][3],iris[1][4][3],iris[1][5][3],iris[1][6][3],iris[1][7][3],iris[1][8][3],iris[1][9][3],iris[1][10][3],iris[1][11][3],iris[1][12][3],iris[1][13][3],iris[1][14][3],iris[1][15][3],iris[1][16][3],iris[1][17][3],iris[1][18][3],iris[1][19][3],iris[1][20][3],iris[1][21][3],iris[1][22][3],iris[1][23][3],iris[1][24][3],iris[1][25][3],iris[1][26][3],iris[1][27][3],iris[1][28][3],iris[1][29][3],iris[1][30][3],iris[1][31][3],iris[1][32][3],iris[1][33][3],iris[1][34][3],iris[1][35][3],iris[1][36][3],iris[1][37][3],iris[1][38][3],iris[1][39][3],iris[1][40][3],iris[1][41][3],iris[1][42][3],iris[1][43][3],iris[1][44][3],iris[1][45][3],iris[1][46][3],iris[1][47][3],iris[1][48][3],iris[1][49][3])
virginica_petal_width=(iris[2][0][3],iris[2][1][3],iris[2][2][3],iris[2][3][3],iris[2][4][3],iris[2][5][3],iris[2][6][3],iris[2][7][3],iris[2][8][3],iris[2][9][3],iris[2][10][3],iris[2][11][3],iris[2][12][3],iris[2][13][3],iris[2][14][3],iris[2][15][3],iris[2][16][3],iris[2][17][3],iris[2][18][3],iris[2][19][3],iris[2][20][3],iris[2][21][3],iris[2][22][3],iris[2][23][3],iris[2][24][3],iris[2][25][3],iris[2][26][3],iris[2][27][3],iris[2][28][3],iris[2][29][3],iris[2][30][3],iris[2][31][3],iris[2][32][3],iris[2][33][3],iris[2][34][3],iris[2][35][3],iris[2][36][3],iris[2][37][3],iris[2][38][3],iris[2][39][3],iris[2][40][3],iris[2][41][3],iris[2][42][3],iris[2][43][3],iris[2][44][3],iris[2][45][3],iris[2][46][3],iris[2][47][3],iris[2][48][3],iris[2][49][3])

# we won't be using this array, it's too disparate petal_widths=(setosa_petal_width, versicolor_petal_width,virginica_petal_width)


fig, ax = plt.subplots()

#Scatter plot
ax.scatter(setosa_petal_width,setosa_petal_length, marker="o",label="Setosa")
ax.scatter(versicolor_petal_width,versicolor_petal_length,marker="d", label="Versicolor")
ax.scatter(virginica_petal_width,virginica_petal_length,marker="v", label="Virginica")

# Labels
ax.set_xlabel("Petal Widths")
ax.set_ylabel("Petal Lengths")
ax.legend()

plt.title("Relationship between Petal Lengths & Widths")
#plt.show()
plt.savefig("petal_lengths_vs_petal_widths.png")


# Use polyfit to fit a line to the data.
m1, c1 = np.polyfit(setosa_petal_width, setosa_petal_length, 1)
m2, c2 = np.polyfit(versicolor_petal_width, versicolor_petal_length, 1)
m3, c3 = np.polyfit(virginica_petal_width, virginica_petal_length, 1)


print(f"\nSetosa:\t\t y = {round(m1,2)}x + {round(c1,2)}")
print(f"Versicolor:\t y = {round(m2,2)}x + {round(c2,2)}")
print(f"Virginica:\t y = {round(m3,2)}x + {round(c3,2)}\n")

# We know what our slope and intercept coefficients are now, we can now create y-coordinates for each value of our petal widths.
print(f"{setosa_petal_width[0]}")
#count = 1
#setosa_petal_width() = setosa_petal_width
#while count <10:
setosa_petal_width[0] = setosa_petal_width[0]* 2 + c1
#count = count + 1
#setosa_petal_width = setosa_petal_width[:]*2.0
print(f"{setosa_petal_width[0]}")
'''
'''

print (f"The y-coordinates for equation of a line for Setosa are: {y1}")


y2 = versicolor_petal_width

count = 0
while count <50:
    y2[count] = (versicolor_petal_width[count] * m2)+c2
    count = count + 1

print (f"The y-coordinates for equation of a line for Versicolor are: {y2}")

y3 = virginica_petal_width

count = 0
while count <50:
    y3[count] = (virginica_petal_width[count] * m3)+c3
    count = count + 1

print (f"The y-coordinates for equation of a line for Virginica are:{y3}")



#The y-coordinates for equation of a line for Setosa are: [np.float64(2.057987837357953), np.float64(2.057987837357953), np.float64(2.057987837357953), np.float64(2.057987837357953), np.float64(2.057987837357953), np.float64(2.107139196474688), np.float64(2.0825635169163204), np.float64(2.057987837357953), np.float64(2.057987837357953), np.float64(2.033412157799585), np.float64(2.057987837357953), np.float64(2.057987837357953), np.float64(2.033412157799585), np.float64(2.033412157799585), np.float64(2.057987837357953), np.float64(2.107139196474688), np.float64(2.107139196474688), np.float64(2.0825635169163204), np.float64(2.0825635169163204), np.float64(2.0825635169163204), np.float64(2.057987837357953), np.float64(2.107139196474688), np.float64(2.057987837357953), np.float64(2.1317148760330555), np.float64(2.057987837357953), np.float64(2.057987837357953), np.float64(2.107139196474688), np.float64(2.057987837357953), np.float64(2.057987837357953), np.float64(2.057987837357953), np.float64(2.057987837357953), np.float64(2.107139196474688), np.float64(2.033412157799585), np.float64(2.057987837357953), np.float64(2.033412157799585), np.float64(2.057987837357953), np.float64(2.057987837357953), np.float64(2.033412157799585), np.float64(2.057987837357953), np.float64(2.057987837357953), np.float64(2.0825635169163204), np.float64(2.0825635169163204), np.float64(2.057987837357953), np.float64(2.1562905555914234), np.float64(2.107139196474688), np.float64(2.0825635169163204), np.float64(2.057987837357953), np.float64(2.057987837357953), np.float64(2.057987837357953), np.float64(2.057987837357953)]
#The y-coordinates for equation of a line for Versicolor are: [np.float64(10.003182424038997), np.float64(10.352619909365766), np.float64(10.352619909365766), np.float64(9.65374493871223), np.float64(10.352619909365766), np.float64(9.65374493871223), np.float64(10.702057394692535), np.float64(8.605432482731924), np.float64(9.65374493871223), np.float64(10.003182424038997), np.float64(8.605432482731924), np.float64(10.352619909365766), np.float64(8.605432482731924), np.float64(10.003182424038997), np.float64(9.65374493871223), np.float64(10.003182424038997), np.float64(10.352619909365766), np.float64(8.605432482731924), np.float64(10.352619909365766), np.float64(8.954869968058693), np.float64(11.40093236534607), np.float64(9.65374493871223), np.float64(10.352619909365766), np.float64(9.30430745338546), np.float64(9.65374493871223), np.float64(10.003182424038997), np.float64(10.003182424038997), np.float64(11.051494880019302), np.float64(10.352619909365766), np.float64(8.605432482731924), np.float64(8.954869968058693), np.float64(8.605432482731924), np.float64(9.30430745338546), np.float64(10.702057394692535), np.float64(10.352619909365766), np.float64(10.702057394692535), np.float64(10.352619909365766), np.float64(9.65374493871223), np.float64(9.65374493871223), np.float64(9.65374493871223), np.float64(9.30430745338546), np.float64(10.003182424038997), np.float64(9.30430745338546), np.float64(8.605432482731924), np.float64(9.65374493871223), np.float64(9.30430745338546), np.float64(9.65374493871223), np.float64(9.65374493871223), np.float64(8.954869968058693), np.float64(9.65374493871223)]
#The y-coordinates for equation of a line for Virginica are:[np.float64(8.032816229254953), np.float64(7.781449431547505), np.float64(7.865238364116655), np.float64(7.739554965262929), np.float64(7.907132830401229), np.float64(7.865238364116655), np.float64(7.697660498978355), np.float64(7.739554965262929), np.float64(7.739554965262929), np.float64(8.032816229254953), np.float64(7.823343897832079), np.float64(7.781449431547505), np.float64(7.865238364116655), np.float64(7.823343897832079), np.float64(7.990921762970379), np.float64(7.949027296685804), np.float64(7.739554965262929), np.float64(7.907132830401229), np.float64(7.949027296685804), np.float64(7.613871566409205), np.float64(7.949027296685804), np.float64(7.823343897832079), np.float64(7.823343897832079), np.float64(7.739554965262929), np.float64(7.865238364116655), np.float64(7.739554965262929), np.float64(7.739554965262929), np.float64(7.739554965262929), np.float64(7.865238364116655), np.float64(7.65576603269378), np.float64(7.781449431547505), np.float64(7.823343897832079), np.float64(7.907132830401229), np.float64(7.613871566409205), np.float64(7.571977100124631), np.float64(7.949027296685804), np.float64(7.990921762970379), np.float64(7.739554965262929), np.float64(7.739554965262929), np.float64(7.865238364116655), np.float64(7.990921762970379), np.float64(7.949027296685804), np.float64(7.781449431547505), np.float64(7.949027296685804), np.float64(8.032816229254953), np.float64(7.949027296685804), np.float64(7.781449431547505), np.float64(7.823343897832079), np.float64(7.949027296685804), np.float64(7.739554965262929)]
    



#fig, ax = plt.subplots()

# Scatter plot
#ax.scatter(setosa_petal_width,setosa_petal_length, marker="o",label="Setosa")
#ax.scatter(versicolor_petal_width,versicolor_petal_length,marker="d", label="Versicolor")
#ax.scatter(virginica_petal_width,virginica_petal_length,marker="v", label="Virginica")
#ax.plot(setosa_petal_width, y1, color='blue')
#ax.plot(versicolor_petal_width, y2, color='orange')
#ax.plot(virginica_petal_width, y3, color='green')

# Labels
#ax.set_xlabel("Petal Widths")
#ax.set_ylabel("Petal Lengths")
#ax.legend()

#plt.title("Relationship between Petal Widths & Lengths")
#plt.show()


'''
# ************************************************************************************************************************************************
# Sepal Lengths vs Sepal Widths (Scatter Plot)
# ***************************************************************************************************************************************************
'''
# from above, we previously calculated the sepal lengths for each class of iris
setosa_sepal_length=(iris[0][0][0],iris[0][1][0],iris[0][2][0],iris[0][3][0],iris[0][4][0],iris[0][5][0],iris[0][6][0],iris[0][7][0],iris[0][8][0],iris[0][9][0],iris[0][10][0],iris[0][11][0],iris[0][12][0],iris[0][13][0],iris[0][14][0],iris[0][15][0],iris[0][16][0],iris[0][17][0],iris[0][18][0],iris[0][19][0],iris[0][20][0],iris[0][21][0],iris[0][22][0],iris[0][23][0],iris[0][24][0],iris[0][25][0],iris[0][26][0],iris[0][27][0],iris[0][28][0],iris[0][29][0],iris[0][30][0],iris[0][31][0],iris[0][32][0],iris[0][33][0],iris[0][34][0],iris[0][35][0],iris[0][36][0],iris[0][37][0],iris[0][38][0],iris[0][39][0],iris[0][40][0],iris[0][41][0],iris[0][42][0],iris[0][43][0],iris[0][44][0],iris[0][45][0],iris[0][46][0],iris[0][47][0],iris[0][48][0],iris[0][49][0])
versicolor_sepal_length=(iris[1][0][0],iris[1][1][0],iris[1][2][0],iris[1][3][0],iris[1][4][0],iris[1][5][0],iris[1][6][0],iris[1][7][0],iris[1][8][0],iris[1][9][0],iris[1][10][0],iris[1][11][0],iris[1][12][0],iris[1][13][0],iris[1][14][0],iris[1][15][0],iris[1][16][0],iris[1][17][0],iris[1][18][0],iris[1][19][0],iris[1][20][0],iris[1][21][0],iris[1][22][0],iris[1][23][0],iris[1][24][0],iris[1][25][0],iris[1][26][0],iris[1][27][0],iris[1][28][0],iris[1][29][0],iris[1][30][0],iris[1][31][0],iris[1][32][0],iris[1][33][0],iris[1][34][0],iris[1][35][0],iris[1][36][0],iris[1][37][0],iris[1][38][0],iris[1][39][0],iris[1][40][0],iris[1][41][0],iris[1][42][0],iris[1][43][0],iris[1][44][0],iris[1][45][0],iris[1][46][0],iris[1][47][0],iris[1][48][0],iris[1][49][0])
virginica_sepal_length=(iris[2][0][0],iris[2][1][0],iris[2][2][0],iris[2][3][0],iris[2][4][0],iris[2][5][0],iris[2][6][0],iris[2][7][0],iris[2][8][0],iris[2][9][0],iris[2][10][0],iris[2][11][0],iris[2][12][0],iris[2][13][0],iris[2][14][0],iris[2][15][0],iris[2][16][0],iris[2][17][0],iris[2][18][0],iris[2][19][0],iris[2][20][0],iris[2][21][0],iris[2][22][0],iris[2][23][0],iris[2][24][0],iris[2][25][0],iris[2][26][0],iris[2][27][0],iris[2][28][0],iris[2][29][0],iris[2][30][0],iris[2][31][0],iris[2][32][0],iris[2][33][0],iris[2][34][0],iris[2][35][0],iris[2][36][0],iris[2][37][0],iris[2][38][0],iris[2][39][0],iris[2][40][0],iris[2][41][0],iris[2][42][0],iris[2][43][0],iris[2][44][0],iris[2][45][0],iris[2][46][0],iris[2][47][0],iris[2][48][0],iris[2][49][0])

# we won't be using this array, it's too disparate petal_lengths=(setosa_sepal_length, versicolor_sepal_length,virginica_sepal_length)

# from above, we previously calculated the sepal widths for each class of iris
setosa_petal_width=(iris[0][0][3],iris[0][1][3],iris[0][2][3],iris[0][3][3],iris[0][4][3],iris[0][5][3],iris[0][6][3],iris[0][7][3],iris[0][8][3],iris[0][9][3],iris[0][10][3],iris[0][11][3],iris[0][12][3],iris[0][13][3],iris[0][14][3],iris[0][15][3],iris[0][16][3],iris[0][17][3],iris[0][18][3],iris[0][19][3],iris[0][20][3],iris[0][21][3],iris[0][22][3],iris[0][23][3],iris[0][24][3],iris[0][25][3],iris[0][26][3],iris[0][27][3],iris[0][28][3],iris[0][29][3],iris[0][30][3],iris[0][31][3],iris[0][32][3],iris[0][33][3],iris[0][34][3],iris[0][35][3],iris[0][36][3],iris[0][37][3],iris[0][38][3],iris[0][39][3],iris[0][40][3],iris[0][41][3],iris[0][42][3],iris[0][43][3],iris[0][44][3],iris[0][45][3],iris[0][46][3],iris[0][47][3],iris[0][48][3],iris[0][49][3])
versicolor_petal_width=(iris[1][0][3],iris[1][1][3],iris[1][2][3],iris[1][3][3],iris[1][4][3],iris[1][5][3],iris[1][6][3],iris[1][7][3],iris[1][8][3],iris[1][9][3],iris[1][10][3],iris[1][11][3],iris[1][12][3],iris[1][13][3],iris[1][14][3],iris[1][15][3],iris[1][16][3],iris[1][17][3],iris[1][18][3],iris[1][19][3],iris[1][20][3],iris[1][21][3],iris[1][22][3],iris[1][23][3],iris[1][24][3],iris[1][25][3],iris[1][26][3],iris[1][27][3],iris[1][28][3],iris[1][29][3],iris[1][30][3],iris[1][31][3],iris[1][32][3],iris[1][33][3],iris[1][34][3],iris[1][35][3],iris[1][36][3],iris[1][37][3],iris[1][38][3],iris[1][39][3],iris[1][40][3],iris[1][41][3],iris[1][42][3],iris[1][43][3],iris[1][44][3],iris[1][45][3],iris[1][46][3],iris[1][47][3],iris[1][48][3],iris[1][49][3])
virginica_petal_width=(iris[2][0][3],iris[2][1][3],iris[2][2][3],iris[2][3][3],iris[2][4][3],iris[2][5][3],iris[2][6][3],iris[2][7][3],iris[2][8][3],iris[2][9][3],iris[2][10][3],iris[2][11][3],iris[2][12][3],iris[2][13][3],iris[2][14][3],iris[2][15][3],iris[2][16][3],iris[2][17][3],iris[2][18][3],iris[2][19][3],iris[2][20][3],iris[2][21][3],iris[2][22][3],iris[2][23][3],iris[2][24][3],iris[2][25][3],iris[2][26][3],iris[2][27][3],iris[2][28][3],iris[2][29][3],iris[2][30][3],iris[2][31][3],iris[2][32][3],iris[2][33][3],iris[2][34][3],iris[2][35][3],iris[2][36][3],iris[2][37][3],iris[2][38][3],iris[2][39][3],iris[2][40][3],iris[2][41][3],iris[2][42][3],iris[2][43][3],iris[2][44][3],iris[2][45][3],iris[2][46][3],iris[2][47][3],iris[2][48][3],iris[2][49][3])
# we won't be using this array, it's too disparate sepal_widths=(setosa_sepal_width, versicolor_sepal_width,virginica_sepal_width)
'''

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
plt.savefig("sepal_lengths_vs_sepal_widths.png")


# Use polyfit to fit a line to the data.
m1, c1 = np.polyfit(setosa_sepal_width, setosa_sepal_length, 1)
m2, c2 = np.polyfit(versicolor_sepal_width, versicolor_sepal_length, 1)
m3, c3 = np.polyfit(virginica_sepal_width, virginica_sepal_length, 1)

print(f"\nSetosa:\t\t y = {round(m1,2)}x + {round(c1,2)}")
print(f"Versicolor:\t y = {round(m2,2)}x + {round(c2,2)}")
print(f"Virginica:\t y = {round(m3,2)}x + {round(c3,2)}\n")


# We know what our slope and intercept coefficients are now, we can now create y-coordinates for each value of our petal widths.
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
plt.show()


#The y-coordinates for equation of a line for Setosa are: [np.float64(2.057987837357953), np.float64(2.057987837357953), np.float64(2.057987837357953), np.float64(2.057987837357953), np.float64(2.057987837357953), np.float64(2.107139196474688), np.float64(2.0825635169163204), np.float64(2.057987837357953), np.float64(2.057987837357953), np.float64(2.033412157799585), np.float64(2.057987837357953), np.float64(2.057987837357953), np.float64(2.033412157799585), np.float64(2.033412157799585), np.float64(2.057987837357953), np.float64(2.107139196474688), np.float64(2.107139196474688), np.float64(2.0825635169163204), np.float64(2.0825635169163204), np.float64(2.0825635169163204), np.float64(2.057987837357953), np.float64(2.107139196474688), np.float64(2.057987837357953), np.float64(2.1317148760330555), np.float64(2.057987837357953), np.float64(2.057987837357953), np.float64(2.107139196474688), np.float64(2.057987837357953), np.float64(2.057987837357953), np.float64(2.057987837357953), np.float64(2.057987837357953), np.float64(2.107139196474688), np.float64(2.033412157799585), np.float64(2.057987837357953), np.float64(2.033412157799585), np.float64(2.057987837357953), np.float64(2.057987837357953), np.float64(2.033412157799585), np.float64(2.057987837357953), np.float64(2.057987837357953), np.float64(2.0825635169163204), np.float64(2.0825635169163204), np.float64(2.057987837357953), np.float64(2.1562905555914234), np.float64(2.107139196474688), np.float64(2.0825635169163204), np.float64(2.057987837357953), np.float64(2.057987837357953), np.float64(2.057987837357953), np.float64(2.057987837357953)]
#The y-coordinates for equation of a line for Versicolor are: [np.float64(10.003182424038997), np.float64(10.352619909365766), np.float64(10.352619909365766), np.float64(9.65374493871223), np.float64(10.352619909365766), np.float64(9.65374493871223), np.float64(10.702057394692535), np.float64(8.605432482731924), np.float64(9.65374493871223), np.float64(10.003182424038997), np.float64(8.605432482731924), np.float64(10.352619909365766), np.float64(8.605432482731924), np.float64(10.003182424038997), np.float64(9.65374493871223), np.float64(10.003182424038997), np.float64(10.352619909365766), np.float64(8.605432482731924), np.float64(10.352619909365766), np.float64(8.954869968058693), np.float64(11.40093236534607), np.float64(9.65374493871223), np.float64(10.352619909365766), np.float64(9.30430745338546), np.float64(9.65374493871223), np.float64(10.003182424038997), np.float64(10.003182424038997), np.float64(11.051494880019302), np.float64(10.352619909365766), np.float64(8.605432482731924), np.float64(8.954869968058693), np.float64(8.605432482731924), np.float64(9.30430745338546), np.float64(10.702057394692535), np.float64(10.352619909365766), np.float64(10.702057394692535), np.float64(10.352619909365766), np.float64(9.65374493871223), np.float64(9.65374493871223), np.float64(9.65374493871223), np.float64(9.30430745338546), np.float64(10.003182424038997), np.float64(9.30430745338546), np.float64(8.605432482731924), np.float64(9.65374493871223), np.float64(9.30430745338546), np.float64(9.65374493871223), np.float64(9.65374493871223), np.float64(8.954869968058693), np.float64(9.65374493871223)]
#The y-coordinates for equation of a line for Virginica are:[np.float64(8.032816229254953), np.float64(7.781449431547505), np.float64(7.865238364116655), np.float64(7.739554965262929), np.float64(7.907132830401229), np.float64(7.865238364116655), np.float64(7.697660498978355), np.float64(7.739554965262929), np.float64(7.739554965262929), np.float64(8.032816229254953), np.float64(7.823343897832079), np.float64(7.781449431547505), np.float64(7.865238364116655), np.float64(7.823343897832079), np.float64(7.990921762970379), np.float64(7.949027296685804), np.float64(7.739554965262929), np.float64(7.907132830401229), np.float64(7.949027296685804), np.float64(7.613871566409205), np.float64(7.949027296685804), np.float64(7.823343897832079), np.float64(7.823343897832079), np.float64(7.739554965262929), np.float64(7.865238364116655), np.float64(7.739554965262929), np.float64(7.739554965262929), np.float64(7.739554965262929), np.float64(7.865238364116655), np.float64(7.65576603269378), np.float64(7.781449431547505), np.float64(7.823343897832079), np.float64(7.907132830401229), np.float64(7.613871566409205), np.float64(7.571977100124631), np.float64(7.949027296685804), np.float64(7.990921762970379), np.float64(7.739554965262929), np.float64(7.739554965262929), np.float64(7.865238364116655), np.float64(7.990921762970379), np.float64(7.949027296685804), np.float64(7.781449431547505), np.float64(7.949027296685804), np.float64(8.032816229254953), np.float64(7.949027296685804), np.float64(7.781449431547505), np.float64(7.823343897832079), np.float64(7.949027296685804), np.float64(7.739554965262929)]
    