# classes for neural network
from Class_FNN import NeuralNetwork
from sklearn.neural_network import MLPRegressor

#other needed classes
from Class_replace_impute_encode import ReplaceImputeEncode
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy  as np
import math

# attribute_map = {
#         'fixed_acidity': [0, (3.8, 14.2), [0, 0]] ,
#         'volatile_acidity': [0, (0.08, 1.1), [0, 0]] ,
#         'citric_acid': [0, (0.0, 1.66), [0, 0]] ,
#         'residual_sugar': [0, (0.6, 65.8), [0, 0]] ,
#         'chlorides': [0, (0.009, 0.346), [0, 0]] ,
#         'free_sulfur_dioxide': [0, (2.0, 289.0), [0, 0]] ,
#         'total_sulfur_dioxide': [0, (9.0, 440.0), [0, 0]] ,
#         'density': [0, (0.98711, 1.03898), [0, 0]] ,
#         'pH': [0, (2.72, 3.82), [0, 0]] ,
#         'sulphates': [0, (0.22, 1.08), [0, 0]] ,
#         'alcohol': [0, (8.0, 14.2), [0, 0]] ,
#         'quality': [0, (1, 10), [0, 0]] }

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


df = pd.read_excel('CreditHistory_Clean.xlsx')

rie = ReplaceImputeEncode(data_map=attribute_map, nominal_encoding=None, \
                          interval_scale = 'std', drop=False, display=True)
encoded_df = rie.fit_transform(df)
varlist = ['good_bad']
X = encoded_df.drop(varlist, axis=1)
y = encoded_df[varlist]
np_y = np.ravel(y) #convert dataframe column to flat array

print("\n******** NEURAL NETWORK ********")
#Neural Network
fnn = MLPRegressor(hidden_layer_sizes=(7, 6), activation='logistic', \
                    solver='lbfgs', max_iter=1000, random_state=12345)
fnn = fnn.fit(X, np_y)
# NeuralNetwork.display_nominal_metrics(fnn, X, y)
NeuralNetwork.display_metrics(fnn, X, y)

# Cross-Validation
network_list = [(3), (11), (5, 4), (6, 5), (7, 6)]
# Scoring for Interval Prediction Neural Networks
score_list = ['neg_mean_squared_error', 'neg_mean_absolute_error']
score_names = ['ASE', 'Mean Abs Error']
for nn in network_list:
    print("\nNetwork: ", nn)
    fnn = MLPRegressor(hidden_layer_sizes=nn, activation='logistic', \
                       solver='lbfgs', max_iter=1000, random_state=12345)
    fnn = fnn.fit(X, np_y)
    for i in range(2):
        scores = cross_val_score(fnn, X, np_y, scoring=score_list[i], cv=10)

    print("{:.<20s}{:>6s}{:>13s}".format("Metric", "Mean", "Std. Dev."))
    for i in range(2):
        mean = math.fabs(scores[i].mean())
        std = scores[i].std()
        print("{:.<20s}{:>7.4f}{:>10.4f}".format(score_names[i], mean, std))



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print("#############################STATS AFTER SPLITTING THE DATA######################################")
# dtree = DecisionTreeClassifier(max_depth=6)
#
# dtree.fit(X_train, y_train)
#
# predictions = dtree.predict(X_test)
#
# print ("Accuracy")
# print (accuracy_score(y_test, predictions))
# mat = confusion_matrix(y_test, predictions)
# print(classification_report(y_test, predictions))


fnn = MLPRegressor(hidden_layer_sizes=(6, 5), activation='logistic', \
                    solver='lbfgs', max_iter=1000, random_state=12345)
fnn = fnn.fit(X_train, y_train)
# NeuralNetwork.display_nominal_metrics(fnn, X, y)
NeuralNetwork.display_metrics(fnn, X_train, y_train)