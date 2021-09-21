import pandas as pd
import numpy as np
import os
import re

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV

from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
# Set this parameter to speed up rendering
pio.renderers.default='iframe'

import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

    
def supervised_models(input_file, input_file_long):    
    #read in files
    df = pd.read_csv(input_file)
    df_long = pd.read_csv(input_file_long)
    
    #separate target column
    X=df.iloc[:,2:-6]
    y=df.iloc[:,-1]
    X_long=df_long.iloc[:,2:-6]
    y_long=df_long.iloc[:,-1]
    
    #uncomment to view correlation dataframe
    #print(pd.DataFrame(abs(X.join(y).corr()['Nuclear']).sort_values(ascending=False)))
    
    #mean imputation
    X = X.fillna(X.mean())
    X_long = X_long.fillna(X_long.mean())
    
    # scaling for model fit via cross validation
    X_norm = StandardScaler().fit_transform(X)
    X_norm_long = StandardScaler().fit_transform(X_long)
    
    
    #short_df parameter optimization
    #SVC grid search
    parameters = {'kernel':('linear', 'rbf', 'poly', 'sigmoid'), 'C':[1, 10, 100], 'class_weight': [None, 'balanced'],
                 'probability': [True, False]}
    svc = SVC()
    clf_svc = GridSearchCV(svc, parameters)
    clf_svc.fit(X_norm, y)
    
    #KNN grid search
    parameters = {'n_neighbors':[i for i in range(10)], 'algorithm': ('auto', 'ball_tree', 'kd_tree', 'brute'), 
                                    'p': [1, 2]}
    knn = KNeighborsClassifier()
    clf_knn = GridSearchCV(knn, parameters)
    clf_knn.fit(X_norm, y)
    
    #LR grid search
    parameters = {'penalty':('l1', 'l2', 'elasticnet', 'none'), 'C': [1, 10, 100], 'fit_intercept': [True, False], 
                                    'class_weight': [None, 'balanced'], 'solver': ('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga')}
    lr = LogisticRegression()
    clf_lr = GridSearchCV(lr, parameters)
    clf_lr.fit(X_norm, y)
    
    #Decision Tree grid search
    parameters = {'max_depth':[i for i in range(10)], 'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'], 
                                    'class_weight': [None, 'balanced'], 'max_features': ('auto', 'sqrt', 'log2', None)}
    dt = DecisionTreeClassifier()
    clf_dt = GridSearchCV(dt, parameters)
    clf_dt.fit(X_norm, y)
    
    #Gradient Boosting Classifier grid search
    parameters = {'loss':['deviance', 'exponential'], 'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3], 'n_estimators': [10, 100, 500], 
                                    'max_depth': [i for i in range(5)]}
    gbc = GradientBoostingClassifier()
    clf_gbc = GridSearchCV(gbc, parameters)
    clf_gbc.fit(X_norm, y)
    
    #Random Forest grid search
    parameters = {'n_estimators': [10, 100, 500], 'bootstrap': [True, False],'class_weight': [None, 'balanced', 'balanced_subsample'],
                                    'max_depth': [i for i in range(5)]}
    rf = RandomForestClassifier()
    clf_rf = GridSearchCV(rf, parameters)
    clf_rf.fit(X_norm, y)
    
    #instantiate models on top gridsearch parameters for each model
    models = [
        DummyClassifier(strategy = 'stratified'),
        DummyClassifier(strategy = 'most_frequent'),
        clf_svc.best_estimator_,
        clf_knn.best_estimator_,
        clf_lr.best_estimator_,
        clf_dt.best_estimator_,
        clf_gbc.best_estimator_,
        clf_rf.best_estimator_
        ]
    
    #fit models on short_df
    Score_list = ['Mean Accuracy Score','Mean Precision Score','Mean Recall Score', 'Mean F1 Score']
    scores = ['accuracy','precision','recall', 'f1']
    model_list = ['Dummy (Stratified)','Dummy (Most Frequent)','SVC','KNN','Logistic Regression','Decision Tree','Gradient Boosting','Random Forest']
    
    results_df = pd.DataFrame()
    results_df['Model'] = model_list
    
    # prepping data and scores for looping through models
    score_data = list(zip(Score_list,scores))
    
    for i,model in enumerate(models):
        # Using 5-fold cv evaluate the different models
        for col,score  in score_data:
            # calculate cv scores for each model on df and long df, returning accuracy and F1 scores
            mean_score = cross_val_score(model, X_norm, y, scoring=score, cv=5, n_jobs=-1)
            results_df.loc[i,col] = np.mean(mean_score)
    #uncomment to view results
    #print(results_df)
    
    # Random Forest scored among the highest.  What are the most important features?
    rf = clf_rf.best_estimator_.fit(X_norm,y)
    importance = rf.feature_importances_
    feature_importance = []
    
    for i,v in enumerate(importance):
        feature_importance.append((v, re.sub(r"\(.*\)","", X.columns[i])))
    sorted_importance = sorted(feature_importance, reverse=True)[:10]
    
    graph_df = pd.DataFrame(sorted_importance, columns=['Importance', 'Feature'])
    fig = px.bar(graph_df, x='Feature', y='Importance')
    
    if not os.path.exists("images"):
        os.mkdir("images")
    fig.write_image("images/short_df_rf_importances.png")
    
    gbc = clf_gbc.best_estimator_.fit(X_norm,y)
    importance = gbc.feature_importances_
    feature_importance = []
    
    for i,v in enumerate(importance):
        feature_importance.append((v, re.sub(r"\(.*\)","", X.columns[i])))
    sorted_importance = sorted(feature_importance, reverse=True)[:10]
    
    graph_df = pd.DataFrame(sorted_importance, columns=['Importance', 'Feature'])
    fig = px.bar(graph_df, x='Feature', y='Importance')
    fig.write_image("images/short_df_gbc_importances.png")
    
    #grid search on long_df
    #SVC grid search
    parameters = {'kernel':('linear', 'rbf', 'poly', 'sigmoid'), 'C':[1, 10, 100], 'class_weight': [None, 'balanced'],
                 'probability': [True, False]}
    svc = SVC()
    clf_svc = GridSearchCV(svc, parameters)
    clf_svc.fit(X_norm_long, y_long)
    
    #KNN grid search
    parameters = {'n_neighbors':[i for i in range(10)], 'algorithm': ('auto', 'ball_tree', 'kd_tree', 'brute'), 
                                    'p': [1, 2]}
    knn = KNeighborsClassifier()
    clf_knn = GridSearchCV(knn, parameters)
    clf_knn.fit(X_norm_long, y_long)
    
    #LR grid search
    parameters = {'penalty':('l1', 'l2', 'elasticnet', 'none'), 'C': [1, 10, 100], 'fit_intercept': [True, False], 
                                    'class_weight': [None, 'balanced'], 'solver': ('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga')}
    lr = LogisticRegression()
    clf_lr = GridSearchCV(lr, parameters)
    clf_lr.fit(X_norm_long, y_long)
    
    #Decision Tree grid search
    parameters = {'max_depth':[i for i in range(10)], 'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'], 
                                    'class_weight': [None, 'balanced'], 'max_features': ('auto', 'sqrt', 'log2', None)}
    dt = DecisionTreeClassifier()
    clf_dt = GridSearchCV(dt, parameters)
    clf_dt.fit(X_norm_long, y_long)
    
    
    #Gradient Boosting Classifier grid search
    parameters = {'loss':['deviance', 'exponential'], 'learning_rate': [0.01, 0.1, 0.2, 0.3], 'n_estimators': [10, 100, 500], 
                                    'max_depth': [i for i in range(5)]}
    gbc = GradientBoostingClassifier()
    clf_gbc = GridSearchCV(gbc, parameters)
    clf_gbc.fit(X_norm_long, y_long)
    
    #Random Forest grid search
    parameters = {'n_estimators': [10, 100, 500], 'bootstrap': [True, False],'class_weight': [None, 'balanced', 'balanced_subsample'],
                                    'max_depth': [i for i in range(5)]}
    rf = RandomForestClassifier()
    clf_rf = GridSearchCV(rf, parameters)
    clf_rf.fit(X_norm_long, y_long)
    
    models = [
        DummyClassifier(strategy = 'stratified'),
        DummyClassifier(strategy = 'most_frequent'),
        clf_svc.best_estimator_,
        clf_knn.best_estimator_,
        clf_lr.best_estimator_,
        clf_dt.best_estimator_,
        clf_gbc.best_estimator_,
        clf_rf.best_estimator_
        ]
    
    Score_list = ['Mean Accuracy Score','Mean Precision Score','Mean Recall Score', 'Mean F1 Score']
    scores = ['accuracy','precision','recall', 'f1']
    model_list = ['Dummy (Stratified)','Dummy (Most Frequent)','SVC','KNN','Logistic Regression','Decision Tree','Gradient Boosting','Random Forest']
    
    results_df_long = pd.DataFrame()
    results_df_long['Model'] = model_list
    
    # prepping data and scores for looping through models
    score_data = list(zip(Score_list,scores))
    
    for i,model in enumerate(models):
        for col,score  in score_data:
            # calculate cv scores for each model on df and long df, returning accuracy and F1 scores
            mean_score = cross_val_score(model, X_norm_long, y_long, scoring=score, cv=5, n_jobs=-1)
            results_df_long.loc[i,col] = np.mean(mean_score)

    return results_df, results_df_long

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input_file', help='the cleaned datafile of nuclear indicators (CSV)')
    parser.add_argument(
        'input_file_long', help='the cleaned long datafile for all indicators (CSV)')
    parser.add_argument(
        'output_file', help='the results of the model using nuclear datafile (CSV)')
    parser.add_argument(
        'output_file_long', help='the results of the model using long datafile (CSV)')
    args = parser.parse_args()

    results, results_long = supervised_models(args.input_file, args.input_file_long)
    results.to_csv(args.output_file, index=False)
    results_long.to_csv(args.output_file_long, index=False)







