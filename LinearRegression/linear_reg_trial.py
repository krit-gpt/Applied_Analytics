import pandas as pd
import numpy  as np
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression

#file_path = '/Users/Home/Desktop/python/CSV/'
df = pd.read_csv("diamondswmissing.csv")
n_obs = df.shape[0]
print("\n********** Data Preprocessing ***********")
print("Data contains %i observations & %i columns.\n" % df.shape)
# df['Price'] = df['Price'].str.replace('$','')
# df['Price'] = df['Price'].astype(dtype='float')
initial_missing = df.isnull().sum()
feature_names = np.array(df.columns.values)
for feature in feature_names:
    if initial_missing[feature] > (n_obs / 2):
        print(feature + ":\n\t%i missing: Drop this attribute." \
              % initial_missing[feature])



# Category Values for Nominal and Binary Attributes
#n_interval = 7
#n_binary = 0
#n_nominal = 4
#n_cat = n_binary + n_nominal


attribute_map = {
    'obs' : [0,(1,53940), [0,0]],
    'Carat': [0, (0.2, 5.5), [0, 0]],
    'cut': [2, ('Fair', 'Good', 'Ideal', 'Premium', 'Very Good'), [0, 0]],
    'color': [2, ('D', 'E', 'F', 'G', 'H', 'I', 'J'), [0, 0]],
    'clarity': [2, ('I1','IF', 'SI1', 'SI2', 'VS1', 'VS2', 'VVS1', 'VVS2'), [0, 0]],
    'depth': [0, (40, 80), [0, 0]],
    'table': [0, (40, 100), [0, 0]],
    'x': [0, (0,11), [0,0]],
    'y' : [0, (0,60), [0,0]],
    'z' : [0, (0,32), [0,0]],
    'price': [0, (300, 20000), [0, 0]]}

# Initialize number missing in attribute_map
for k, v in attribute_map.items():
    for feature in feature_names:
        if feature == k:
            v[2][0] = initial_missing[feature]
            break
'''
print("#################################### MIN/MEAN/MAX #############################################")
for k,v in attribute_map.items():
    if v[0] == 0:
        print("For", k, " Min is", min(df[k]))
        print("For", k, " Max is", max(df[k]))
        print("For", k, " Mean is", df[k].mean())
'''

# Scan for outliers among interval attributes
nan_map = df.isnull()

for i in range(n_obs):
    # Check for outliers in interval attributes
    for k, v in attribute_map.items():
        if nan_map.loc[i, k] == True:
            continue
        if v[0] == 0:  # Interval Attribute
            l_limit = v[1][0]
            u_limit = v[1][1]
            if df.loc[i, k] > u_limit or df.loc[i, k] < l_limit:
                v[2][1] += 1
                df.loc[i, k] = None
        else:  # Categorical Attribute
            in_cat = False
            for cat in v[1]:
                if df.loc[i, k] == cat:
                    in_cat = True
            if in_cat == False:
                df.loc[i, k] = None
                v[2][1] += 1

print("\nNumber of missing values and outliers by attribute:")
feature_names = np.array(df.columns.values)
for k, v in attribute_map.items():
    print(k + ":\t%i missing" % v[2][0] + "  %i outlier(s)" % v[2][1])

interval_attributes = []
nominal_attributes = []
binary_attributes = []
onehot_attributes = []
for k, v in attribute_map.items():
    if v[0] == 0:
        interval_attributes.append(k)
    else:
        if v[0] == 1:
            binary_attributes.append(k)
        else:
            nominal_attributes.append(k)
            for i in range(len(v[1])):
                str = k + ("%i" % i)
                onehot_attributes.append(str)

n_interval = len(interval_attributes)
n_binary = len(binary_attributes)
n_nominal = len(nominal_attributes)
n_onehot = len(onehot_attributes)
print("\nFound %i Interval Attribute(s), " % n_interval, \
      "%i Binary," % n_binary, \
      "and %i Nominal Attribute(s)\n" % n_nominal)

# print("Original DataFrame:\n", df[0:5])
# Put the interval data from the dataframe into a numpy array
interval_data = df.as_matrix(columns=interval_attributes)
# Create the Imputer for the Interval Data
interval_imputer = preprocessing.Imputer(strategy='mean')
# Impute the missing values in the Interval data
imputed_interval_data = interval_imputer.fit_transform(interval_data)

# Convert String Categorical Attribute to Numbers
# Create a dictionary with mapping of categories to numbers for attribute 'B'
cat_map = {'Good': 1, 'Very Good': 2, 'Fair': 3, 'Ideal': 4, 'Premium' : 5}
# Change the string categories of 'B' to numbers
df['cut'] = df['cut'].map(cat_map)
cat_map = {'D': 1, 'E': 2, 'F': 3, 'G': 4, 'H': 5, 'I': 6, 'J': 7}

df['color'] = df['color'].map(cat_map)
cat_map = {'IF': 1, 'SI1': 2, 'SI2': 3, 'VS1': 4, 'VS2': 5, 'VVS1': 6, 'VVS2': 7, 'I1': 8}
# Change the string categories of 'B' to numbers
df['clarity'] = df['clarity'].map(cat_map)
#cat_map = {'AGS': 1, 'GIA': 2}
# Change the string categories of 'B' to numbers
#df['report'] = df['report'].map(cat_map)

# Put the nominal and binary data from the dataframe into a numpy array
nominal_data = df.as_matrix(columns=nominal_attributes)
#binary_data = df.as_matrix(columns=binary_attributes)
# Create Imputer for Categorical Data
cat_imputer = preprocessing.Imputer(strategy='most_frequent')
# Impute the missing values in the Categorical Data
imputed_nominal_data = cat_imputer.fit_transform(nominal_data)
#imputed_binary_data = cat_imputer.fit_transform(binary_data)

# Encoding Interval Data by Scaling
scaler = preprocessing.StandardScaler()  # Create an instance of StandardScaler()
scaler.fit(imputed_interval_data)
scaled_interval_data = scaler.transform(imputed_interval_data)

# Create an instance of the OneHotEncoder & Selecting Attributes
onehot = preprocessing.OneHotEncoder()
hot_array = onehot.fit_transform(imputed_nominal_data).toarray()

# Bring Interval and Categorial Data Together
# The Imputed Data
data_array = np.hstack((imputed_interval_data, \
                        imputed_nominal_data))
col = []
for i in range(n_interval):
    col.append(interval_attributes[i])
#for i in range(n_binary):
   # col.append(binary_attributes[i])
for i in range(n_nominal):
    col.append(nominal_attributes[i])
df_imputed = pd.DataFrame(data_array, columns=col)
print("\nImputed DataFrame:\n", df_imputed[0:15])

# The Imputed and Encoded Data
data_array = np.hstack((scaled_interval_data, hot_array))
# col = (interval_attributes, cat_attributes)
col = []
for i in range(n_interval):
    col.append(interval_attributes[i])
#for i in range(n_binary):
    #col.append(binary_attributes[i])
for i in range(n_onehot):
    col.append(onehot_attributes[i])
df_imputed_scaled = pd.DataFrame(data_array, columns=col)
df_imputed_scaled = df_imputed_scaled.drop(['cut0', 'color0', 'clarity0'], axis=1)
print("\nImputed & Scaled DataFrame:\n", df_imputed_scaled[0:15])

print("\n********** Linear Regression ************")
print("****** Predicting Diamond Price *********")
lr = LinearRegression()
target = np.asarray(df['price'])
X = np.asarray(df_imputed_scaled.drop('price', axis=1))
lr.fit(X, target)
print("Linear Regression with one-hot encoding:")
print("Intercept:", lr.intercept_)
col_dropped = col
col_dropped.remove('price')
col_dropped.remove('cut0')
col_dropped.remove('color0')
col_dropped.remove('clarity0')
for i in range(X.shape[1]):
    print("Coef - " + col_dropped[i] + ":", lr.coef_[i])
print(X)


'''
################################  WITHOUT ONE HOT ENCODING -- WRONG ##############################################
lr = LinearRegression()
X = np.asarray(df_imputed.drop('price', axis=1))
lr.fit(X, target)
print("\nLinear Regression without one-hot encoding: This is incorrect!")
print("Intercept:", lr.intercept_)
col = ['carat', 'depth', 'table', 'report', 'cut', 'color', 'clarity']
for i in range(X.shape[1]):
    print("Coef - " + col[i] + ":", lr.coef_[i])
'''
