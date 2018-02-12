from Class_replace_impute_encode import ReplaceImputeEncode
from Class_regression import logreg
import pandas as pd
import numpy  as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score



df = pd.read_excel('credithistory_HW2.xlsx')


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
    'telephon':[1,(1,2),[0,0]]}


rie = ReplaceImputeEncode(data_map=attribute_map, display=True)
encoded_df = rie.fit_transform(df)

y = np.asarray(encoded_df['good_bad']) # The target is not scaled or imputed
X = np.asarray(encoded_df.drop('good_bad', axis=1))
lgr = LogisticRegression()
lgr.fit(X, y)
logreg.display_coef(lgr, X, y, rie.col)
logreg.display_binary_metrics(lgr, X, y)

X_train, X_validate, y_train, y_validate = \
            train_test_split(X,y,test_size = 0.3, random_state=7)
lgr_train = LogisticRegression()
lgr_train.fit(X_train, y_train)
print("\nTraining Data\nRandom Selection of 70% of Original Data")
logreg.display_binary_split_metrics(lgr_train, X_train, y_train, \
                                    X_validate, y_validate)

lgr_4_scores = cross_val_score(lgr, X_train, y_train, cv=4)
print("\nAccuracy Scores by Fold: ", lgr_4_scores)
print("Accuracy Mean:      %.4f" %lgr_4_scores.mean())
print("Accuracy Std. Dev.: %.4f" %lgr_4_scores.std())