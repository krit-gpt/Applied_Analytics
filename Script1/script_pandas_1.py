# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 20:39:18 2018

@author: Neehar
"""

print("Howdy from within my iPython Console")

obs1 = [1, 2, 3]
obs2 = [3, 4, 5]
data = [obs1, obs2]

var_avg = []
for var in range(3):
    sum = 0
    for obs in data:
        sum = sum + obs[var]
    avg = sum/len(data)
    var_avg.append(avg) 
print(var_avg) 

students = {}
students['a1'] = ['Peter', 85]
students['b2'] = ['Paul' , 90]
students['x3'] = ['Mary' , 95]
print(students['a1'][0])

s2017 = {}
s2017['UCLA']           = [44, 45]
s2017['Nicholls']       = [24, 14]
s2017['Lafayette']      = [45, 21]
s2017['Arkansas']       = [50, 43]
s2017['South Carolina'] = [24, 17]
s2017['Alabama']        = [19, 27]
s2017['Florida']        = [19, 17]
s2017['MS State']       = [14, 35]
s2017['Auburn']         = [27, 42]
s2017['New Mexico']     = [55, 14]
s2017['Mississippi']    = [31, 24]
s2017['LSU']            = [21, 45]
n_games = len(s2017)
tamu_points  = 0
other_points = 0
for game in s2017:
    tamu_points  += s2017[game][0]
    other_points += s2017[game][1]
tamu_avg  = tamu_points /n_games
other_avg = other_points/n_games
print("*Average Points/Game*")
print("Games 2017:\t   %i" %n_games)
print("Texas A&M:\t %.1f" %tamu_avg, "\nOpponents:\t %.1f" %other_avg)


import numpy as np

vector = np.array([1, 2, 3, 4])
# print the dimensions of this array
print("Array Dimensions:", vector.ndim)

help(np)
print("Array Contents:  ", vector)
# print the 2nd element in the vector
print(" is the 2nd element %.1f of this vector" %vector[1])

array = np.array([[1, 2, 3], [4, 5, 6]])
# print the dimensions of this array
print("Array Dimensions:", array.ndim)
# print the array
print("Array Contents:\n", array)
print("This array has %i rows and %i columns." %array.shape)
# print the contents of the 2nd row, 1st column.
print("The 2nd row, 1st column of this array contains %.1f" %array[1][0])

import pandas as pd

array = np.array([[1, 2, 3], [4, 5, 6]])
df = pd.DataFrame(array, columns=['X1', 'X2', 'Y'], index=['1st', '2nd'])
print(df)

import os
os.chdir(file_path)

import pandas as pd
file_path = 'E:\Python Projects'
sonar_df = pd.read_csv(file_path+'\sonar_hw1.csv')
print(sonar_x)
sonar_x = sonar_df.iloc[0:, 0:60]
sonar_y = sonar_df['object']
# Print first 4 observations - frequencies and object
print("Frequencies:\n", sonar_x[0:4], "\n\nObject Detected:")
print(sonar_y[0:4])

table = pd.DataFrame(index=sonar_new.columns.values, columns=['Min','Max','Median','Null','n_high','n_low'])

table['Min'] = sonar_new.min()
table['Max'] = sonar_new.max()
table['Median'] = sonar_new.median()
table['Null'] = sonar_x.isnull().sum(axis=0)
table['n_low'] = sonar_x[sonar_x[:]<0].count() + sonar_x.isnull().sum(axis=0)
table['n_high'] = sonar_x[sonar_x[:]>1].count()

sonar = sonar_x[sonar_x[:]>0]
sonar_new = sonar[sonar[:]<1]

print(table)

from pandas import ExcelWriter
file_path = 'E:\Python Projects'
writer = ExcelWriter('HW1.xlsx')
table.to_excel(writer)
writer.save()

sonar_x.min()


print(sonar_df[0:60].median(), '\t',  np.max(sonar_x), np.min(sonar_x), sonar_df.isnull().sum(axis=0))
print(np.max(sonar_x))
print(np.min(sonar_x))
print(sonar_df.isnull().sum(axis=0)) #axis = 0 gives everything column wise

print(len(sonar_df.columns))
z=len(sonar_df.columns)-1
print(z)
print(sonar_x)
lis=[];

count1= []
count2= []
for j in range(z):
    count3 = 0
    count4 = 0
    for i in range(len(sonar_x)):
        if sonar_x[i,j] < 0 :
            count3+=1;
        if sonar_x[i,j] > 1:
            count4+=1;     
    count1.append(count3)
    count2.append(count4)
print(count1, '\t' , count2, '\t')    
        
        
lis.append(sonar_df.iloc[i,j])
print(sonar_df.iloc[1,2])
print(pd.
