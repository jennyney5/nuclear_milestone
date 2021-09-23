# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import re
import pickle

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

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering

from sklearn import metrics
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

def unsupervised_models(input_file):    
    #read in files
    df = pd.read_csv(input_file)
    
    #separate target column
    X=df.iloc[:,2:-6]
    y=df.iloc[:,-1]

    
    #uncomment to view correlation dataframe
    #print(pd.DataFrame(abs(X.join(y).corr()['Nuclear']).sort_values(ascending=False)))
    
    #mean imputation
    X = X.fillna(X.mean())

    
    # scaling for model fit via cross validation
    X_norm = StandardScaler().fit_transform(X)
    X_scaled = MinMaxScaler().fit_transform(X)

    # Kmeans clustering
    sil = {}
    db = {}
    ch = {}
    
    for k in range(2,10):
        km = KMeans(n_clusters = k, init='k-means++', max_iter=100, n_init=1, random_state=42)
        km.fit(X_scaled)
        sil[k] = metrics.silhouette_score(X_scaled, km.labels_, sample_size=5000)
        db[k] = metrics.davies_bouldin_score(X_scaled, km.labels_)
        ch[k] = metrics.calinski_harabasz_score(X_scaled, km.labels_)
    # High Silhouette score returns k=2
    # High Calinksi-Harabasz return k=2 and low Davies-Bouldin returns k=3

    k=2
    km = KMeans(n_clusters = k, init='k-means++', max_iter=100, n_init=1,random_state = 42)
    km.fit(X_scaled)
    top_indices = (-km.cluster_centers_).argsort()[:,:1]
    Top_Terms = np.take(X.columns,top_indices)
    #consider output Top Terms to json
        
    #Apply clusters to countries
    clusters = km.predict(X_scaled)

    #plotX is a DataFrame of scaled X
    plotX = pd.DataFrame(np.array(X_scaled))
    
    #Rename plotX's columns since it was briefly converted to an np.array above
    plotX.columns = X.columns
    plotX['Cluster'] = clusters
    
    plotX['Cluster'] = plotX['Cluster'].astype('str')
    plotX['Nuclear'] = y
    plotX['Nuclear'] = plotX['Nuclear'].astype('str')
    
    #PCA with one principal component
    pca_1d = PCA(n_components=1, random_state = 42)
    
    #PCA with two principal components
    pca_2d = PCA(n_components=2, random_state = 42)
    
    #PCA with three principal components
    pca_3d = PCA(n_components=3, random_state = 42)
    
    #This DataFrame holds that single principal component mentioned above
    PCs_1d = pd.DataFrame(pca_1d.fit_transform(plotX.drop(["Cluster","Nuclear"], axis=1)))
    
    #This DataFrame contains the two principal components that will be used
    #for the 2-D visualization mentioned above
    PCs_2d = pd.DataFrame(pca_2d.fit_transform(plotX.drop(["Cluster","Nuclear"], axis=1)))
    
    #And this DataFrame contains three principal components that will aid us
    #in visualizing our clusters in 3-D
    PCs_3d = pd.DataFrame(pca_3d.fit_transform(plotX.drop(["Cluster","Nuclear"], axis=1)))
    
    PCs_1d.columns = ["PC1_1d"]
    
    #"PC1_2d" means: 'The first principal component of the components created for 2-D visualization, by PCA.'
    #And "PC2_2d" means: 'The second principal component of the components created for 2-D visualization, by PCA.'
    PCs_2d.columns = ["PC1_2d", "PC2_2d"]
    
    PCs_3d.columns = ["PC1_3d", "PC2_3d", "PC3_3d"]
    
    plotX = pd.concat([plotX,PCs_1d,PCs_2d,PCs_3d], axis=1, join='inner')
    plotX["dummy"] = 0    
    
    fig = px.scatter(plotX, x="PC1_1d", y="PC2_2d",color="Cluster",facet_col="Nuclear", title = "Kmeans Clustering visualized by 2 principal components")
    if not os.path.exists("images"):
        os.mkdir("images")
    fig.write_image("images/short_df_k2Cluster_PCA2.png")
    
    # Spectral clustering
    sp = SpectralClustering(n_clusters = 2,affinity='nearest_neighbors',random_state = 42)
    sp.fit(X_scaled)
    clusters = sp.labels_
    
    #plotX is a DataFrame of scaled X
    plotX_s = pd.DataFrame(np.array(X_scaled))
    
    #Rename plotX's columns since it was briefly converted to an np.array above
    plotX_s.columns = X.columns
    plotX_s['Cluster'] = clusters
    
    plotX_s['Cluster'] = plotX_s['Cluster'].astype('str')
    plotX_s['Nuclear'] = y
    plotX_s['Nuclear'] = plotX_s['Nuclear'].astype('str')
        
    #PCA with one principal component
    pca_1d = PCA(n_components=1, random_state = 42)
    
    #PCA with two principal components
    pca_2d = PCA(n_components=2, random_state = 42)
    
    #PCA with three principal components
    pca_3d = PCA(n_components=3, random_state = 42)
    
    #This DataFrame holds that single principal component mentioned above
    PCs_1d = pd.DataFrame(pca_1d.fit_transform(plotX_s.drop(["Cluster","Nuclear"], axis=1)))
    
    #This DataFrame contains the two principal components that will be used
    #for the 2-D visualization mentioned above
    PCs_2d = pd.DataFrame(pca_2d.fit_transform(plotX_s.drop(["Cluster","Nuclear"], axis=1)))
    
    #And this DataFrame contains three principal components that will aid us
    #in visualizing our clusters in 3-D
    PCs_3d = pd.DataFrame(pca_3d.fit_transform(plotX_s.drop(["Cluster","Nuclear"], axis=1)))
    
    PCs_1d.columns = ["PC1_1d"]
    
    #"PC1_2d" means: 'The first principal component of the components created for 2-D visualization, by PCA.'
    #And "PC2_2d" means: 'The second principal component of the components created for 2-D visualization, by PCA.'
    PCs_2d.columns = ["PC1_2d", "PC2_2d"]
    
    PCs_3d.columns = ["PC1_3d", "PC2_3d", "PC3_3d"]
    
    plotX_s = pd.concat([plotX_s,PCs_1d,PCs_2d,PCs_3d], axis=1, join='inner')
    plotX_s["dummy"] = 0    
    
    fig = px.scatter(plotX_s, x="PC1_1d", y="PC2_2d",color="Cluster",facet_col="Nuclear", title = "Spectral Clustering visualized by 2 principal components")
    fig.write_image("images/short_df_Spectral_PCA2.png")
    
    
    # PCA Dimensionality reduction
    pca = PCA(n_components = 8,random_state = 0).fit(X_norm)
    X_pca = pca.transform(X_norm)
    df_pca = pd.DataFrame({'Explained Variance Ratio':pca.explained_variance_ratio_,
                   'PC #':[1,2,3,4,5,6,7,8]})
    fig = px.bar(df_pca, x="PC #", y="Explained Variance Ratio",title = "Explained Variance Ratio by Principal Components")
    fig.write_image("images/short_df_PCA8_Exp_Variance.png")
    
    principal_df = pd.DataFrame(data = X_pca
             , columns = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8'])
    final_df = pd.concat([principal_df, y], axis = 1)
    final_df['Nuclear'] = final_df['Nuclear'].astype(str)
    fig = px.scatter(final_df, x="PC1", y="PC2",color="Nuclear", title = "First 2 Principal Components")
    fig.write_image("images/short_df_PCA8_First2PC.png")
    
    
    
    #short_df parameter optimization
    #SVC grid search
    parameters = {'kernel':('linear', 'rbf', 'poly', 'sigmoid'), 'C':[1, 10, 100], 'class_weight': [None, 'balanced'],
                 'probability': [True, False]}
    svc = SVC()
    clf_svc = GridSearchCV(svc, parameters)
    clf_svc.fit(X_pca, y)
    
    #KNN grid search
    parameters = {'n_neighbors':[i for i in range(10)], 'algorithm': ('auto', 'ball_tree', 'kd_tree', 'brute'), 
                                    'p': [1, 2]}
    knn = KNeighborsClassifier()
    clf_knn = GridSearchCV(knn, parameters)
    clf_knn.fit(X_pca, y)
    
    #LR grid search
    parameters = {'penalty':('l1', 'l2', 'elasticnet', 'none'), 'C': [1, 10, 100], 'fit_intercept': [True, False], 
                                    'class_weight': [None, 'balanced'], 'solver': ('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga')}
    lr = LogisticRegression()
    clf_lr = GridSearchCV(lr, parameters)
    clf_lr.fit(X_pca, y)
    
    #Decision Tree grid search
    parameters = {'max_depth':[i for i in range(10)], 'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'], 
                                    'class_weight': [None, 'balanced'], 'max_features': ('auto', 'sqrt', 'log2', None)}
    dt = DecisionTreeClassifier()
    clf_dt = GridSearchCV(dt, parameters)
    clf_dt.fit(X_pca, y)
    
    #Gradient Boosting Classifier grid search
    parameters = {'loss':['deviance', 'exponential'], 'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3], 'n_estimators': [10, 100, 500], 
                                    'max_depth': [i for i in range(5)]}
    gbc = GradientBoostingClassifier()
    clf_gbc = GridSearchCV(gbc, parameters)
    clf_gbc.fit(X_pca, y)
    
    #Random Forest grid search
    parameters = {'n_estimators': [10, 100, 500], 'bootstrap': [True, False],'class_weight': [None, 'balanced', 'balanced_subsample'],
                                    'max_depth': [i for i in range(5)]}
    rf = RandomForestClassifier()
    clf_rf = GridSearchCV(rf, parameters)
    clf_rf.fit(X_pca, y)
    
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
       
    return results_df, models
    

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input_file', help='the cleaned datafile of nuclear indicators (CSV)')
    parser.add_argument(
        'output_file', help='the results of the unsupervised model after dimensionality reduction (CSV)')
    parser.add_argument(
        'output_file_models', help='the best estimators model of the unsupervised model after dimensionality reduction (PKL)')
    args = parser.parse_args()
    
    results, list_models = unsupervised_models(args.input_file)
    results.to_csv(args.output_file, index=False)
    
    with open(args.output_file_models, "wb") as f:
        for model in list_models:
            pickle.dump(model, f)
