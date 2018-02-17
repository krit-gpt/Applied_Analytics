Stat 656 - Homework Week 4 Solution
This assignment consists of two parts. You can choose which part you would like to do. Part 1 involves using SAS Enterprise Miner and Part 2 uses Python. You are required to one of these, but you can do both if you like. You'll receive extra credit for doing both parts.

Both parts use the same data file DiamondsWMissing.csv available for download from the Week 3 eCampus assignment directory.

In both parts, the target is a binary classification target labeled good_bad in the data. Use the data dictionary supplied with this assignment on eCampus to identify outliers. Replace them with missing values and then impute all missing values.

Part 1 - SAS Enterprise Miner
In Part 1, imputation uses the SAS Tree method for both interval and categorical variables. SAS refers to categorical variables as class variables.

After identifying all outliers, setting them to missing and then imputing all missing values, fit a logistic regression model to these data using good_bad as the target. Report the model metrics available from the Class Class_regression.logreg

Split the data into two parts, 70% for training and 30% for validation.

Find and evaluate the best model comparing the HP and Non-HP logistic regression solutions determined using stepwise model selection. Identify the best model of these three. Note - these methods for hyperparameter optimization, stepwise, are not available in python.

Report the best model and print the min, max and mean of the predicted and actual amounts, and the ASE for the training and validation models. Also print the frequency tables for each categorical attribute.

Finally calcuate and report the sensitifity, specificity, precision, accuracy and F1-score for the best model.

Part 2 - Python
Use the same data, except you do not have stepwise, forward and backward regression. As a result, just fit the logistic regression model with all attributes in the data. Use one-hot encoding, dropping the last one-hot column for each attribute. Use the mean for imputing missing values for interval attributes and the mode for imputing all others.

First print the min, max and mean of the predicted and actual prices, and the ASE for the training and validation models. Also print the frequency tables for the categorical attributes, and the metrics described in class - sensitivity specifity, precision, accuracy and f1-score.

Step 1: Read the data using the pandas read for csv files
The key thing is to get the file path correct. Different systems use different paths to your project directory. In linux and Mac OS, the path separator symbol is a forward slash /. In windows it is a backslash \. However in python you cannot use a single backslash to represent the path sepator. Python requires you use a double backslash to which is translated into a single backslash. The reason for this is because a single backslash is also used to represent important unprintable characters such as a line feed \n and a tab \t.

The following code reads the data, but it also has the names of key packages needed with this solution. The last two classes were developed for this course to make data preprocessing and routine printing a little easier. Make sure you have the two Python Class files loaded in the same directory you are using for your python project code.

In [1]:
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#  classes provided for the course
from Class_replace_impute_encode import ReplaceImputeEncode
from Class_regression import logreg
    
print("***** Stat 656 Week 4 Homework ***")
file_path = '/Users/Home/Desktop/python/Excel/'
df = pd.read_excel(file_path+"credithistory.xlsx")
df=df.drop('purpose', axis=1) # purpose has more than 50% missing
***** Stat 656 Week 4 Homework ***
Step 2: Create you data map
The data map is some of your metadata. It describes your attributes, their level (interval, binary or nominal), and the characteristics of that attributes.

Here is the data map constructed from the data dictionary for this data file. Notice that the number '3' is used for any attribute that you don't want changed. Usually that would be your target, as well as other columns that contain case identifications or labels.

In [2]:
# First Integer Designates Data Type
# 0=Interval, 1=Binary, 2=Nominal, 3=Other (No Changes, do not include)
attribute_map = {
    'age':[0,(1, 120),[0,0]],
    'amount':[0,(0, 20000),[0,0]],
    'duration':[0,(1,100),[0,0]],
    'checking':[2,(1, 2, 3, 4),[0,0]],
    'coapp':[2,(1,2,3),[0,0]],
    'depends':[1,(1,2),[0,0]],
    'employed':[2,(1,2,3,4,5),[0,0]],
    'existcr':[2,(1,2,3,4),[0,0]],
    'foreign':[1,(1,2),[0,0]],
    'good_bad':[1,('bad', 'good'),[0,0]],
    'history':[2,(0,1,2,3,4),[0,0]],
    'installp':[2,(1,2,3,4),[0,0]],
    'job':[2,(1,2,3,4),[0,0]],
    'marital':[2,(1,2,3,4),[0,0]],
    'other':[2,(1,2,3),[0,0]],
    'property':[2,(1,2,3,4),[0,0]],
 #   'purpose':[1,(0,1,2,3,4,5,6,7,8,9,'X'),[0,0]],
    'resident':[2,(1,2,3,4),[0,0]],
    'savings':[2,(1,2,3,4,5),[0,0]],
    'telephon':[1,(1,2),[0,0]] }
Step 3: Replace-Impute-Encode
Next, use the class ReplaceImputeEncode() to replace outliers with missing values, impute missing values and then scale interval data and encode categorial data.

The ReplaceImputeEncode() class allows you to specify None for scaling and/or encoding. It also lets you select 'one-hot' or 'SAS' encoding for categorical variables. In most other software this is automatic, but for Python we need to setup our own scaling and encoding.

The complete API for this class is described in the class. First you instantiate the class then you use fit_transform() to actually process your dataframe.

In [3]:
encoding = 'SAS' # Categorical encoding:  Use 'SAS', 'one-hot' or None
scale    = None  # Interval scaling:  Use 'std', 'robust' or None
scaling  = 'No'  # Text description for interval scaling

rie = ReplaceImputeEncode(data_map=attribute_map, nominal_encoding=encoding, \
                          interval_scale = scale, display=True)

#features_map = rie.draft_features_map(df)
encoded_df = rie.fit_transform(df)
********** Data Preprocessing ***********
Features Dictionary Contains:
3 Interval, 
4 Binary, and 
12 Nominal Attribute(s).

Data contains 1000 observations & 20 columns.


Attribute Counts
............... Missing  Outliers
age.......        35         6
amount....        12         9
duration..        42         0
checking..         0         0
coapp.....        12         0
depends...         0         0
employed..         0         6
existcr...         0         0
foreign...         0         0
good_bad..         0         0
history...         0         0
installp..         0         0
job.......         0         0
marital...         9         5
other.....         0         0
property..         0         0
resident..        11         0
savings...         4         2
telephon..        19         0
Step 4: Fit a linear regression model
Now fit a linear regression model to the entire dataframe

In [4]:
y = np.asarray(encoded_df['good_bad']) # The target is not scaled or imputed
X = np.asarray(encoded_df.drop('good_bad',axis=1))
lgr = LogisticRegression()
lgr.fit(X, y)
Out[4]:
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)
Step 5: Print Model Metrics
You can print model metrics using the logreg class found in the python file Class_regression. Ensure that file is placed into your python project directory.

In [5]:
print("\nLogistic Regression with\n\t" + encoding + " encoding and " + \
      scaling + " scaling.")
print("\nLogistic Regression Model using Entire Dataset")
col = rie.col
col.remove('good_bad')
logreg.display_coef(lgr, 43, 2, col)
logreg.display_binary_metrics(lgr, X, y)
Logistic Regression with
	SAS encoding and No scaling.

Logistic Regression Model using Entire Dataset

Coefficients:
Intercept..        -1.1586
age........        -0.0171
amount.....         0.0001
duration...         0.0247
depends....        -0.1359
foreign....         0.4375
telephon...         0.1126
checking0..         0.7859
checking1..         0.3536
checking2..        -0.1133
coapp0.....         0.1564
coapp1.....         0.5932
employed0..         0.0556
employed1..         0.3355
employed2..         0.0517
employed3..        -0.3805
existcr0...        -0.4889
existcr1...        -0.0662
existcr2...        -0.0379
history0...         0.5103
history1...         0.6104
history2...        -0.0185
history3...        -0.2069
installp0..        -0.3356
installp1..        -0.1923
installp2..         0.0781
job0.......         0.0065
job1.......         0.0442
job2.......         0.0492
marital0...         0.3630
marital1...         0.1696
marital2...        -0.4332
other0.....         0.2280
other1.....         0.1017
property0..        -0.2953
property1..        -0.0135
property2..        -0.1196
resident0..        -0.3923
resident1..         0.3189
resident2..         0.0772
savings0...         0.5063
savings1...         0.3133
savings2...         0.1520
savings3...        -0.5939

Model Metrics
Observations...............      1000
Coefficients...............        44
DF Error...................       956
Mean Absolute Error........    0.3060
Avg Squared Error..........    0.1515
Accuracy...................    0.7750
Precision..................    0.6667
Recall (Sensitivity).......    0.5000
F1-Score...................    0.5714
MISC (Misclassification)...     22.5%
     class -1..............     10.7%
     class 1...............     50.0%


     Confusion
       Matrix     Class -1  Class 1  
Class -1....       625        75
Class 1.....       150       150
Step 6: Evalute the model using Simple Cross Validation
Here we split the data into two parts, a training and validation part consisting of 70% and 30% of the data, respectively.

In [6]:
X_train, X_validate, y_train, y_validate = \
            train_test_split(X,y,test_size = 0.3, random_state=7)
lgr_train = LogisticRegression()
lgr_train.fit(X_train, y_train)
print("\nTraining Data\nRandom Selection of 70% of Original Data")
logreg.display_binary_split_metrics(lgr_train, X_train, y_train, \
                                     X_validate, y_validate)
Training Data
Random Selection of 70% of Original Data


Model Metrics..........       Training     Validation
Observations...........            700            300
Coefficients...........             44             44
DF Error...............            656            256
Mean Absolute Error....         0.3138         0.3054
Avg Squared Error......         0.1553         0.1506
Accuracy...............         0.7629         0.7767
Precision..............         0.6624         0.6212
Recall (Sensitivity)...         0.4793         0.4940
F1-score...............         0.5561         0.5503
MISC (Misclassification)...      23.7%          22.3%
     class -1..............      11.0%          11.5%
     class 1...............      52.1%          50.6%


Training
Confusion Matrix  Class -1  Class 1  
Class -1....       430        53
Class 1.....       113       104


Validation
Confusion Matrix  Class -1  Class 1  
Class -1....       192        25
Class 1.....        42        41
Entire Code
The following is a listing of all parts described above.

In [7]:
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#  classes provided for the course
from Class_replace_impute_encode import ReplaceImputeEncode
from Class_regression import logreg
    
print("***** Stat 656 Week 4 Homework ***")
file_path = '/Users/Home/Desktop/python/Excel/'
df = pd.read_excel(file_path+"credithistory.xlsx")
df=df.drop('purpose', axis=1)

# First Integer Designates Data Type
# 0=Interval, 1=Binary, 2=Nominal, 3=Other (No Changes, do not include)
attribute_map = {
    'age':[0,(1, 120),[0,0]],
    'amount':[0,(0, 20000),[0,0]],
    'duration':[0,(1,100),[0,0]],
    'checking':[2,(1, 2, 3, 4),[0,0]],
    'coapp':[2,(1,2,3),[0,0]],
    'depends':[1,(1,2),[0,0]],
    'employed':[2,(1,2,3,4,5),[0,0]],
    'existcr':[2,(1,2,3,4),[0,0]],
    'foreign':[1,(1,2),[0,0]],
    'good_bad':[1,('bad', 'good'),[0,0]],
    'history':[2,(0,1,2,3,4),[0,0]],
    'installp':[2,(1,2,3,4),[0,0]],
    'job':[2,(1,2,3,4),[0,0]],
    'marital':[2,(1,2,3,4),[0,0]],
    'other':[2,(1,2,3),[0,0]],
    'property':[2,(1,2,3,4),[0,0]],
 #   'purpose':[1,(0,1,2,3,4,5,6,7,8,9,'X'),[0,0]],
    'resident':[2,(1,2,3,4),[0,0]],
    'savings':[2,(1,2,3,4,5),[0,0]],
    'telephon':[1,(1,2),[0,0]] }

encoding = 'SAS' # Categorical encoding:  Use 'SAS', 'one-hot' or None
scale    = None  # Interval scaling:  Use 'std', 'robust' or None
scaling  = 'No'  # Text description for interval scaling

rie = ReplaceImputeEncode(data_map=attribute_map, nominal_encoding=encoding, \
                          interval_scale = scale, display=True)

#features_map = rie.draft_features_map(df)
encoded_df = rie.fit_transform(df)

y = np.asarray(encoded_df['good_bad']) # The target is not scaled or imputed
X = np.asarray(encoded_df.drop('good_bad',axis=1))
lgr = LogisticRegression()
lgr.fit(X, y)

print("\nLogistic Regression with\n\t" + encoding + " encoding and " + \
      scaling + " scaling.")
print("\nLogistic Regression Model using Entire Dataset")
col = rie.col
col.remove('good_bad')
logreg.display_coef(lgr, 43, 2, col)
logreg.display_binary_metrics(lgr, X, y)

X_train, X_validate, y_train, y_validate = \
            train_test_split(X,y,test_size = 0.3, random_state=7)
lgr_train = LogisticRegression()
lgr_train.fit(X_train, y_train)
print("\nTraining Data\nRandom Selection of 70% of Original Data")
logreg.display_binary_split_metrics(lgr_train, X_train, y_train, \
                                     X_validate, y_validate)
***** Stat 656 Week 4 Homework ***

********** Data Preprocessing ***********
Features Dictionary Contains:
3 Interval, 
4 Binary, and 
12 Nominal Attribute(s).

Data contains 1000 observations & 20 columns.


Attribute Counts
............... Missing  Outliers
age.......        35         6
amount....        12         9
duration..        42         0
checking..         0         0
coapp.....        12         0
depends...         0         0
employed..         0         6
existcr...         0         0
foreign...         0         0
good_bad..         0         0
history...         0         0
installp..         0         0
job.......         0         0
marital...         9         5
other.....         0         0
property..         0         0
resident..        11         0
savings...         4         2
telephon..        19         0

Logistic Regression with
	SAS encoding and No scaling.

Logistic Regression Model using Entire Dataset

Coefficients:
Intercept..        -1.1586
age........        -0.0171
amount.....         0.0001
duration...         0.0247
depends....        -0.1359
foreign....         0.4375
telephon...         0.1126
checking0..         0.7859
checking1..         0.3536
checking2..        -0.1133
coapp0.....         0.1564
coapp1.....         0.5932
employed0..         0.0556
employed1..         0.3355
employed2..         0.0517
employed3..        -0.3805
existcr0...        -0.4889
existcr1...        -0.0662
existcr2...        -0.0379
history0...         0.5103
history1...         0.6104
history2...        -0.0185
history3...        -0.2069
installp0..        -0.3356
installp1..        -0.1923
installp2..         0.0781
job0.......         0.0065
job1.......         0.0442
job2.......         0.0492
marital0...         0.3630
marital1...         0.1696
marital2...        -0.4332
other0.....         0.2280
other1.....         0.1017
property0..        -0.2953
property1..        -0.0135
property2..        -0.1196
resident0..        -0.3923
resident1..         0.3189
resident2..         0.0772
savings0...         0.5063
savings1...         0.3133
savings2...         0.1520
savings3...        -0.5939

Model Metrics
Observations...............      1000
Coefficients...............        44
DF Error...................       956
Mean Absolute Error........    0.3060
Avg Squared Error..........    0.1515
Accuracy...................    0.7750
Precision..................    0.6667
Recall (Sensitivity).......    0.5000
F1-Score...................    0.5714
MISC (Misclassification)...     22.5%
     class -1..............     10.7%
     class 1...............     50.0%


     Confusion
       Matrix     Class -1  Class 1  
Class -1....       625        75
Class 1.....       150       150

Training Data
Random Selection of 70% of Original Data


Model Metrics..........       Training     Validation
Observations...........            700            300
Coefficients...........             44             44
DF Error...............            656            256
Mean Absolute Error....         0.3138         0.3054
Avg Squared Error......         0.1553         0.1506
Accuracy...............         0.7629         0.7767
Precision..............         0.6624         0.6212
Recall (Sensitivity)...         0.4793         0.4940
F1-score...............         0.5561         0.5503
MISC (Misclassification)...      23.7%          22.3%
     class -1..............      11.0%          11.5%
     class 1...............      52.1%          50.6%


Training
Confusion Matrix  Class -1  Class 1  
Class -1....       430        53
Class 1.....       113       104


Validation
Confusion Matrix  Class -1  Class 1  
Class -1....       192        25
Class 1.....        42        41