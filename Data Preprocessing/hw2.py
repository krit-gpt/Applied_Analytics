import pandas as pd
import xlrd
import numpy as np
from sklearn import preprocessing
import scipy

df = pd.read_excel('credithistory_HW2.xlsx')


'''
#Calculating the dimensions of the given data --
# df.shape[0] -- number of rows,
# df.shape[1] -- number of cols
# df.shape -- gives a tuple with (rows,cols) -- in this example -- (1000,21)
'''
n_obs = df.shape[0]
n_cols = df.shape[1]
print(n_cols)
#print(df)


# Calculating the number of NANs and putting them in a dictionary!
missing = df.isnull().sum()
feature_names = np.array(df.columns.values)
'''
#make it a numpy array -- df.columns.values !!!
# df.columns.values gives a list of all the columns --
# ['checking' 'duration' 'history' 'purpose' 'amount' 'savings' 'employed'
# 'installp' 'marital' 'coapp' 'resident' 'property' 'age' 'other'
# 'housing' 'existcr' 'job' 'depends' 'telephon' 'foreign' 'good_bad']
'''

'''
Drop the feature with missing > 50% of the number of observations
-- missing = dict with {name: missing_count},
-- features_names is a numpy array of the names of all the features!
-- n_obs was obtained with df.shape[0] -- number of rows!
-- column dropped with df.drop([column_name], axis = 1, inplace = True)
'''
for feature in feature_names:
    if missing[feature] > (n_obs)//2:
        df.drop([feature], axis=1, inplace=True)
        print("Dropped Feature ,", feature)

'''
Make a dictionary of the MetaData now!
# The first number of 0=Interval, 1=Binary, 2=Nominal, and 3=Other (do not encode)
# The 1st tuple for interval attributes is their lower and upper bounds
# The 1st tuple for categorical attributes is their allowed categories
# The 2nd tuple contains the number missing and number of outliers
'''
n_interval = 2
n_binary = 2
n_nominal = 5 - n_interval - n_binary
n_cat = n_binary + n_nominal

'''
attribute_map = {
    'age':[0, (1,120), [0,0]],
    'amount': [0, (0,20000), [0,0]],
    'checking': [3, ('1','2','3','4'),[0,0]],
    'coapp': [2 ,('1','2','3'),[0,0]],
    'depends': [1, ('1','2'), [0,0]],
    'duration': [0, (0,72), [0,0]],
    'employed': [2,('1','2','3','4','5'), [0,0]],
    'existcr': [2,('1','2','3','4'), [0,0]],
    'foreign': [1, ('1','2'), [0,0]],
    'good_bad': [1, ('good', 'bad'), [0,0]],
    'history': [2, ('0','1','2','3','4'), [0,0]],
    'housing': [2, ('1', '2', '3'), [0,0]],
    'installp': [3, ('1','2','3','4'), [0,0]],
    'job': [3, ('1','2','3','4'), [0,0]],
    'marital': [2, ('1','2','3','4'), [0,0]],
    'other': [2, ('1','2','3'), [0,0]],
    'property': [2,('1','2','3','4'), [0,0]],
    'purpose': [2, ('0','1','2','3','4','5','6','7','8','9','10'),[0,0]],
    'resident':[2, ('1','2','3','4'),[0,0]],
    'savings': [3, ('1','2','3','4','5'), [0,0]],
    'telephon': [1, ('1','2'), [0,0]]
}

'''

attribute_map = {
    'age':[0,(1,120),[0,0]],
    'amount':[0,(0,20000),[0,0]],
    'duration':[0,(0,100),[0,0]],
    'checking':[3,(1,2,3,4),[0,0]],
    'coapp':[2,(1,2,3),[0,0]],
    'depends':[1,(1,2),[0,0]],
    'employed':[2,(1,2,3,4,5),[0,0]],
    'existcr':[2,(1,2,3,4),[0,0]],
    'foreign':[1,(1,2),[0,0]],
    'good_bad':[1,('bad','good'),[0,0]],
    'history':[2,(0,1,2,3,4),[0,0]],
    'housing':[2,(1,2,3),[0,0]],
    'installp':[3,(1,2,3,4),[0,0]],
    'job':[3,(1,2,3,4),[0,0]],
    'marital':[2,(1,2,3,4),[0,0]],
    'other':[2,(1,2,3),[0,0]],
    'property':[2,(1,2,3,4),[0,0]],
    'resident':[2,(1,2,3,4),[0,0]],
    'savings':[3,(1,2,3,4,5),[0,0]],
    'telephon':[1,(1,2),[0,0]] }

'''
Putting in the number of missing
'''

for k,v in attribute_map.items():
    for feature in feature_names:
        if feature == k:
            v[2][0] = missing[feature]
            break
#print(attribute_map)


'''
Finding the outliers now -- 
1. Make a nan_map of the missing values -- is a boolean matrix!  
'''

nan_map = df.isnull()
#print(nan_map)

for i in range(n_obs):
    for k,v in attribute_map.items():
        if nan_map.loc[i,k] == True: # it is None
            continue

        if v[0] == 0: #Interval column
            l_limit = v[1][0]  # get lower limit from metadata
            u_limit = v[1][1]  # get upper limit from metadata
            # If the observation is outside the limits, its an outlier
            if df.loc[i, k] > u_limit or df.loc[i, k] < l_limit:
                v[2][1] += 1  # Number of outliers in metadata
                df.loc[i, k] = None  # Set outlier to missing

        else:  # Categorical Attribute or Other
            in_cat = False
            # Iterate over the allowed categories for this attribute
            for cat in v[1]:
                if df.loc[i, k] == cat:  # Found the category, not outlier
                    in_cat = True
            if in_cat == False:  # Did not find this category in the metadata
                df.loc[i, k] = None  # This data is not recognized, its an outlier
                v[2][1] += 1  # Increment the outlier counter for this attribute

print(attribute_map)


'''
DOING HOT-ENCODING -- NEED TO UNDERSTAND!!!

'''

# Each of these lists will contain the names of the attributes in their level
feature_names = np.array(df.columns.values)
interval_attributes = []
nominal_attributes = []
binary_attributes = []
onehot_attributes = []
ordinal_attributes = []
# Iterate over the data dictionary
for k, v in attribute_map.items():
    if v[0] == 0:  # This is an interval attribute
        interval_attributes.append(k)
    else:
        if v[0] == 1:  # This is a binary attribute
            binary_attributes.append(k)
        else:  # Anything else is nominal or other
            if v[0] == 2:  # Other, not treated

                nominal_attributes.append(k)
            # Nominal attributes receive one-hot encoding
            # Generate their special binary columns
                for i in range(len(v[1])):
                    str = k + ("%i" % i)
                    onehot_attributes.append(str)
            else:
                ordinal_attributes.append(k)

n_interval = len(interval_attributes)
n_binary = len(binary_attributes)
n_nominal = len(nominal_attributes)
n_ordinal = len(ordinal_attributes)
n_onehot = len(onehot_attributes)

'''
onehot_attributes == 
coapp0, coapp1, coapp2, employed0, employed1, employed2, employed3, employed4...
'''
print("\nFound %i Interval Attributes, " % n_interval, \
      "%i Binary," % n_binary, \
      "and %i Nominal Attribute\n" % n_nominal)

'''
Imputing the interval attributes -- we have the interval attributes in the list --
interval_attribute in the previous step.
'''

# Put the interval data from the dataframe into a numpy array
interval_data = df.as_matrix(columns=interval_attributes)
# Create the Imputer for the Interval Data
interval_imputer = preprocessing.Imputer(strategy='mean')
# Impute the missing values in the Interval data
imputed_interval_data = interval_imputer.fit_transform(interval_data)
print("Imputed Interval Data:\n", imputed_interval_data)


'''
Convert String categorical attribute to numbers, so that sklearn preprocessing 
can be used to impute categorical with the most frequent category.
'''

# Convert String Categorical Attribute to Numbers
# Create a dictionary with mapping of categories to numbers for attribute 'B'
cat_map = {'good':1, 'bad':2}
cat_map2={'1':1.0, '2':2.0, '3':3.0}
cat_map3={'1':1.0,'2':2.0, '3':3.0, '4':4.0, '5':5.0}
cat_map4={'1':1.0, '2':2.0, '3':3.0, '4':4.0}
cat_map5={'1':1.0, '2':2.0, '3':3.0, '4':4.0}
cat_map6={'0':0.0, '1':1.0, '2':2.0, '3':3.0, '4':4.0}
cat_map7={'1':1.0, '2':2.0, '3':3.0, '4':4.0}
cat_map8={'1':1.0, '2':2.0, '3':3.0, '4':4.0}
cat_map9={'1':1.0, '2':2.0, '3':3.0}
cat_map10={'1':1.0, '2':2.0, '3':3.0, '4':4.0}
#cat_map6={}
#cat_map7={}

# Change the string categories of 'B' to numbers
df['good_bad'] = df['good_bad'].map(cat_map)
# df['coapp'] = df['coapp'].map(cat_map2)
# df['employed'] = df['employed'].map(cat_map3)
# df['existcr'] = df['existcr'].map(cat_map4)
# df['housing'] = df['housing'].map(cat_map5)
# df['history'] = df['history'].map(cat_map6)
# df['marital'] = df['marital'].map(cat_map7)
# df['property'] = df['property'].map(cat_map8)
# df['other'] = df['other'].map(cat_map9)
# df['resident'] = df['resident'].map(cat_map10)

print(df)

'''
Imputing the values of the categorical data with the most frequent
using preprocessing.
'''


# Put the nominal and binary data from the dataframe into a numpy array
nominal_data = df.as_matrix(columns=nominal_attributes)
binary_data  = df.as_matrix(columns=binary_attributes)
ordinal_data = df.as_matrix(columns=ordinal_attributes)

# Create Imputer for Categorical Data
cat_imputer = preprocessing.Imputer(strategy='most_frequent')
# Impute the missing values in the Categorical Data
imputed_nominal_data = cat_imputer.fit_transform(nominal_data)
imputed_binary_data  = cat_imputer.fit_transform(binary_data)
imputed_ordinal_data = cat_imputer.fit_transform(ordinal_data)


#print(df)
'''
- Scaling the interval variables
- Converting to 'z' scores, in order to scale the interval variables!
'''

scaler = preprocessing.StandardScaler() # Create an instance of StandardScaler()
scaler.fit(imputed_interval_data)
scaled_interval_data = scaler.transform(imputed_interval_data)
print("Imputed & Scaled Interval Data\n", scaled_interval_data)


# Create an instance of the OneHotEncoder & Selecting Attributes
onehot = preprocessing.OneHotEncoder()
hot_array = onehot.fit_transform(imputed_nominal_data).toarray()
print(hot_array)

#print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print(len(imputed_binary_data[1]) + len(imputed_interval_data[1]) + len(imputed_nominal_data[1]))

data_array= np.hstack((imputed_interval_data, imputed_binary_data, imputed_nominal_data, imputed_ordinal_data))

print(data_array.shape)

col = []
for i in range(n_interval):
    col.append(interval_attributes[i])
for i in range(n_binary):
    col.append(binary_attributes[i])
for i in range(n_nominal):
    col.append(nominal_attributes[i])
for i in range(n_ordinal):
    col.append(ordinal_attributes[i])

df_imputed = pd.DataFrame(data_array,columns=col)
print("\nImputed DataFrame:\n\n", df_imputed)

# The Imputed and Encoded Data
data_array = np.hstack((scaled_interval_data, imputed_binary_data, imputed_ordinal_data, hot_array))
print(hot_array[1])
#col = (interval_attributes, cat_attributes)
col = []
for i in range(n_interval):
    col.append(interval_attributes[i])
for i in range(n_binary):
    col.append(binary_attributes[i])
for i in range(n_ordinal):
    col.append(ordinal_attributes[i])
for i in range(n_onehot):
    col.append(onehot_attributes[i])
df_imputed_scaled = pd.DataFrame(data_array,columns=col)
print("\nImputed,Scaled & Encoded DataFrame.", df_imputed_scaled)


