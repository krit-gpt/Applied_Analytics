# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 21:38:08 2018

@author: Neehar
"""

import os #Setting file path
import numpy as np
import pandas as pd
file_path = 'E:\Python Projects\HW2'
os.chdir(file_path)
df = pd.read_excel(file_path+'\credithistory_HW2.xlsx') #Reading xlsx
print("Data contains %i observations & %i columns " %df.shape)

n_obs=df.shape[0]
initial_missing = df.isnull().sum()
feature_names = np.array(df.columns.values)
for feature in feature_names:
    if initial_missing[feature]>(n_obs/2):
        df.drop([feature],axis=1,inplace=True)
        print(feature+":\t%i missing: Dropping this attribute." \
                  %initial_missing[feature])



attribute_map = {
    'age':[0,(1,120),[0,0]],
    'amount':[0,(0,20000),[0,0]],
    'duration':[0,(0,100),[0,0]],
    'checking':[2,(1,2,3,4),[0,0]],
    'coapp':[2,(1,2,3),[0,0]],
    'depends':[1,(1,2),[0,0]],
    'employed':[2,(1,2,3,4,5),[0,0]],
    'existcr':[2,(1,2,3,4),[0,0]],
    'foreign':[1,(1,2),[0,0]],
    'good_bad':[1,('bad','good'),[0,0]],
    'history':[2,(0,1,2,3,4),[0,0]],
    'housing':[2,(1,2,3),[0,0]],
    'installp':[2,(1,2,3,4),[0,0]],
    'job':[2,(1,2,3,4),[0,0]],
    'marital':[2,(1,2,3,4),[0,0]],
    'other':[2,(1,2,3),[0,0]],
    'property':[2,(1,2,3,4),[0,0]], 
    'resident':[2,(1,2,3,4),[0,0]],
    'savings':[2,(1,2,3,4,5),[0,0]],
    'telephon':[1,(1,2),[0,0]] }

for k,v in attribute_map.items():
    for feature in feature_names:
        if feature==k:
            v[2][0] = initial_missing[feature]
            break
        

nan_map = df.isnull()
print(nan_map.shape)
for i in range(n_obs):
    # Check for outliers in interval attributes
    for k, v in attribute_map.items():
        if nan_map.loc[i,k]==True:
            continue
        if v[0]==0: # Interval Attribute
            l_limit = v[1][0]
            u_limit = v[1][1]
            if df.loc[i, k]>u_limit or df.loc[i,k]<l_limit:
                v[2][1] += 1
                df.loc[i,k] = None
        else: # Categorical Attribute
            in_cat = False
            for cat in v[1]:
                if df.loc[i,k]==cat:
                    in_cat=True
            if in_cat==False:
                df.loc[i,k] = None
                v[2][1] += 1
                
    
print("\nNumber of missing values and outliers by attribute:")
feature_names = np.array(df.columns.values)
for k,v in attribute_map.items():
    print(k+":\t%i missing" %v[2][0]+ "  %i outlier(s)" %v[2][1])

# Each of these lists will contain the names of the attributes in their level
interval_attributes = []
nominal_attributes  = []
binary_attributes   = []
onehot_attributes   = []
# Iterate over the data dictionary
for k,v in attribute_map.items():
    if v[0]==0: # This is an interval attribute
        interval_attributes.append(k)
    else:
        if v[0]==1: # This is a binary attribute
            binary_attributes.append(k)
        else:       # Anything else is nominal or other
            if v[0]>2: # Other, not treated
                continue
            nominal_attributes.append(k)
            # Nominal attributes receive one-hot encoding
            # Generate their special binary columns
            for i in range(len(v[1])):
                str = k+("%i" %i)
                onehot_attributes.append(str)
            
n_interval = len(interval_attributes)
n_binary   = len(binary_attributes)
n_nominal  = len(nominal_attributes)
n_onehot   = len(onehot_attributes)
print("\nFound %i Interval Attributes, " %n_interval, \
      "%i Binary," %n_binary,  \
      "and %i Nominal Attribute\n" %n_nominal)

all_missing = df.isnull().sum()
feature_names = np.array(df.columns.values)
for feature in feature_names:
    if all_missing[feature]>0:
        print(feature+":\t%i missing" %initial_missing[feature]+ \
        "  %i outlier(s)" %(all_missing[feature]-initial_missing[feature]))

from sklearn import preprocessing

# Put the interval data from the dataframe into a numpy array
interval_data = df.as_matrix(columns=interval_attributes)
# Create the Imputer for the Interval Data
interval_imputer = preprocessing.Imputer(strategy='mean')
# Impute the missing values in the Interval data
imputed_interval_data = interval_imputer.fit_transform(interval_data)
print("Imputed Interval Data:\n", imputed_interval_data)

# Convert String Categorical Attribute to Numbers
# Create a dictionary with mapping of categories to numbers for attribute 'good_bad'
cat_map = {'good':1, 'bad':0}     
# Change the string categories of 'B' to numbers 
df['good_bad'] = df['good_bad'].map(cat_map)


nominal_data = df.as_matrix(columns=nominal_attributes)
binary_data  = df.as_matrix(columns=binary_attributes)
# Create Imputer for Categorical Data
cat_imputer = preprocessing.Imputer(strategy='most_frequent')
# Impute the missing values in the Categorical Data
imputed_nominal_data = cat_imputer.fit_transform(nominal_data)
imputed_binary_data  = cat_imputer.fit_transform(binary_data)
print("Imputed Nominal Data\n", imputed_nominal_data)
print("Imputed Binary Data\n", imputed_binary_data)

#Transforming interval attributes

scaler = preprocessing.StandardScaler() # Create an instance of StandardScaler()
scaler.fit(imputed_interval_data)
scaled_interval_data = scaler.transform(imputed_interval_data)
print("Imputed & Scaled(Transformed) Interval Data\n", scaled_interval_data)

# Create an instance of the OneHotEncoder & Selecting Attributes
onehot = preprocessing.OneHotEncoder()
hot_array = onehot.fit_transform(imputed_nominal_data).toarray()
print("One-Hot Encoding for Nominal Attributes :\n", hot_array)

len(hot_array[1])

# Bring Interval and Categorial Data Together
# The Imputed Data
data_array= np.hstack((imputed_interval_data, imputed_binary_data, imputed_nominal_data))
data_array.shape
col = []
for i in range(n_interval):
    col.append(interval_attributes[i])
for i in range(n_binary):
    col.append(binary_attributes[i])
for i in range(n_nominal):
    col.append(nominal_attributes[i])
df_imputed = pd.DataFrame(data_array,columns=col)
print("\nImputed DataFrame:\n\n", df_imputed)

# The Imputed and Encoded Data
data_array = np.hstack((scaled_interval_data, imputed_binary_data, hot_array))
hot_array[1]
#col = (interval_attributes, cat_attributes)
col = []
for i in range(n_interval):
    col.append(interval_attributes[i])
for i in range(n_binary):
    col.append(binary_attributes[i])
for i in range(n_onehot):
    col.append(onehot_attributes[i])
df_imputed_scaled = pd.DataFrame(data_array,columns=col)
print("\nImputed,Scaled & Encoded DataFrame.", df_imputed_scaled)

from pandas import ExcelWriter
file_path = 'E:\Python Projects'
writer = ExcelWriter('HW2-encoded.xlsx')  #Writing as excel file
df_imputed.to_excel(writer)
writer.save()

