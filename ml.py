import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import precision_score,recall_score,accuracy_score,r2_score,f1_score

def d_tree_regression(data,random_val,test_sz):
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_sz, random_state = 0)

    regressor = DecisionTreeRegressor(random_state = random_val)
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)
    return (r2_score(y_test, y_pred))

def poly_regression(data,test_sz,deg):
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_sz, random_state = 0)

    poly_reg = PolynomialFeatures(degree = deg)
    X_poly = poly_reg.fit_transform(X_train)
    regressor = LinearRegression()
    regressor.fit(X_poly, y_train)

    y_pred = regressor.predict(poly_reg.transform(X_test))
    return (r2_score(y_test, y_pred))

def multi_regression(data,test_sz):
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_sz, random_state = 0)\

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)
    return (r2_score(y_test, y_pred))

def random_forest_regression(data,random_val,test_sz,estimators):
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_sz, random_state = 0)

    regressor = RandomForestRegressor(n_estimators = estimators, random_state = random_val)
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)
    return (r2_score(y_test, y_pred))

def svr(data,kernl,test_sz,c_params,e_params):
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    y = y.reshape(len(y),1)
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X = sc_X.fit_transform(X)
    y = sc_y.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_sz, random_state = 0)
  

    regressor = SVR(kernel = kernl,C=c_params,epsilon=e_params)
    regressor.fit(X_train, y_train)
    
    y_pred = regressor.predict(X_test)

    return r2_score(y_test, y_pred)


def kernel_svm(data,test_sz,krnl,state):
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_sz, random_state = 0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    classifier = SVC(kernel = krnl, random_state = state)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    return accuracy_score(y_test, y_pred),precision_score(y_test,y_pred,pos_label=2),recall_score(y_test,y_pred,pos_label=2),f1_score(y_test,y_pred,pos_label=2)

def logistic(data,test_sz,state):
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_sz, random_state = 0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    classifier = LogisticRegression(random_state = state)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    return accuracy_score(y_test, y_pred),precision_score(y_test,y_pred,pos_label=2),recall_score(y_test,y_pred,pos_label=2),f1_score(y_test,y_pred,pos_label=2)

def random_forest(data,test_sz,state,estimators, criteria):
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_sz, random_state = 0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    classifier = RandomForestClassifier(n_estimators = estimators, criterion = criteria, random_state = state)
    classifier.fit(X_train, y_train)


    y_pred = classifier.predict(X_test)
    return accuracy_score(y_test, y_pred),precision_score(y_test,y_pred,pos_label=2),recall_score(y_test,y_pred,pos_label=2),f1_score(y_test,y_pred,pos_label=2)

def naive_bayes(data,test_sz):
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_sz, random_state = 0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    return accuracy_score(y_test, y_pred),precision_score(y_test,y_pred,pos_label=2),recall_score(y_test,y_pred,pos_label=2),f1_score(y_test,y_pred,pos_label=2)

def KNN(data,test_sz,neighbors):
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_sz, random_state = 0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    classifier = KNeighborsClassifier(n_neighbors = neighbors, metric = 'minkowski', p = 2)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    return accuracy_score(y_test, y_pred),precision_score(y_test,y_pred,pos_label=2),recall_score(y_test,y_pred,pos_label=2),f1_score(y_test,y_pred,pos_label=2)

def d_tree_classification(data,test_sz,state,criteria):
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_sz, random_state = 0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    classifier = DecisionTreeClassifier(criterion = criteria, random_state = state)
    classifier.fit(X_train, y_train)


    y_pred = classifier.predict(X_test)
    return accuracy_score(y_test, y_pred),precision_score(y_test,y_pred,pos_label=2),recall_score(y_test,y_pred,pos_label=2),f1_score(y_test,y_pred,pos_label=2)

def linear_disc(data,test_sz,solver_params):
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_sz, random_state = 0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    classifier = LinearDiscriminantAnalysis(solver =solver_params)
    classifier.fit(X_train, y_train)


    y_pred = classifier.predict(X_test)
    return accuracy_score(y_test, y_pred),precision_score(y_test,y_pred,pos_label=2),recall_score(y_test,y_pred,pos_label=2),f1_score(y_test,y_pred,pos_label=2)

performance_classification = pd.DataFrame(columns=[
    'Model', 'Accuracy_Training_Set', 'Accuracy_Test_Set', 'Precision',
    'Recall', 'f1_score'
])

def performance_metrics_classification(data,model,test_sz, i):

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_sz, random_state = 0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    model_name = type(model).__name__
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    performance_classification.loc[i] = [
        model_name,
        accuracy_score(y_train, y_pred_train),
        accuracy_score(y_test, y_pred_test),
        precision_score(y_test, y_pred_test,pos_label=2),
        recall_score(y_test, y_pred_test,pos_label=2),
        f1_score(y_test, y_pred_test,pos_label=2),
    ]

models_list = [LogisticRegression(),
               DecisionTreeClassifier(),
               RandomForestClassifier(),
               SVC(),
               KNeighborsClassifier(),
               GaussianNB(),LinearDiscriminantAnalysis()
               ]

def classification(data,test_sz):
    for n, model in enumerate(models_list):
                performance_metrics_classification(data,model,test_sz, n)
    return performance_classification




