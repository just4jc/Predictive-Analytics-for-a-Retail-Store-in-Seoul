# -*- coding: utf-8 -*-
"""
Created on Sat May  5 17:47:54 2018

@author: ojhadiwesh
"""
import pandas as pd
import numpy as np

#importing the csv file from my desktop
korea_data= pd.read_csv('D:\PA\Korea data.csv')
korea_data= korea_data.iloc[:, korea_data.columns!='Date']
#impute outlook column by the most occuring type of day which happens to be sunny
outlook= korea_data['Outlook']
outlook= outlook.fillna('Sunny')
korea_data['Outlook']= outlook

#this gives us a correlation martix of all the features
corr= korea_data.corr()

#labelling the data to make it more meaningful and useful for further analysis
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
def labelling(data):
    labelencoder_data = LabelEncoder()
    data['Weekday'] = labelencoder_data.fit_transform(data['Weekday'])
    data['Outlook'] = labelencoder_data.fit_transform(data['Outlook'])
    data['Distance from Station X(Meter)'] = labelencoder_data.fit_transform(data['Distance from Station X(Meter)'])
    data['Distance from Station X(Feet)'] = labelencoder_data.fit_transform(data['Distance from Station X(Feet)'])
    data['Distance from Station Y(Meter)'] = labelencoder_data.fit_transform(data['Distance from Station Y(Meter)'])
    data['Distance from Station Y(Feet)'] = labelencoder_data.fit_transform(data['Distance from Station Y(Feet)'])
    data['Distance from Main Street(Meter)'] = labelencoder_data.fit_transform(data['Distance from Main Street(Meter)'])
    data['Distance from Main Street(Feet)'] = labelencoder_data.fit_transform(data['Distance from Main Street(Feet)'])

labelling(korea_data)

# Dividing the dataset into each store data using store name 
storeA_data= korea_data.loc[korea_data['Store Name']=='Store A']
storeB_data= korea_data.loc[korea_data['Store Name']=='Store B']
storeC_data= korea_data.loc[korea_data['Store Name']=='Store C']
storeD_data= korea_data.loc[korea_data['Store Name']=='Store D']
storeE_data= korea_data.loc[korea_data['Store Name']=='Store E']

#drop store name and store code
storeA_data= storeA_data.loc[:, '# of Customers':'Japanese Tourists']
storeB_data= storeB_data.loc[:, '# of Customers':'Japanese Tourists']
storeC_data= storeC_data.loc[:, '# of Customers':'Japanese Tourists']
storeD_data= storeD_data.loc[:, '# of Customers':'Japanese Tourists']
storeE_data= storeE_data.loc[:, '# of Customers':'Japanese Tourists']

#Imputing the japanese tourists data by filling the mean of the tourist data
def touristimpute(data):
    data['Japanese Tourists']= data['Japanese Tourists'].fillna(data['Japanese Tourists'].mean())

touristimpute(storeA_data)
touristimpute(storeB_data)
touristimpute(storeC_data)
touristimpute(storeD_data)
touristimpute(storeE_data)

#creating correlation matrices for each store
corelationA= storeA_data.corr(method='pearson')
corelationB= storeB_data.corr(method='pearson')
corelationC= storeC_data.corr(method='pearson')
corelationD= storeD_data.corr(method='pearson')
corelationE= storeE_data.corr(method='pearson')


#create train and test data for modeling the data using kfolds cross validation
from sklearn.cross_validation import  cross_val_score
from sklearn.model_selection import KFold

k_folds= KFold(n_splits= 10, shuffle= True, random_state= 0)
for train_index, test_index in k_folds.split(storeA_data):
    storeA_data_train, storeA_data_test = storeA_data.iloc[train_index], storeA_data.iloc[test_index]

k_folds= KFold(n_splits= 10, shuffle= True, random_state= 0)
for train_index, test_index in k_folds.split(storeB_data):
    storeB_data_train, storeB_data_test = storeB_data.iloc[train_index], storeB_data.iloc[test_index]

k_folds= KFold(n_splits= 10, shuffle= True, random_state= 0)
for train_index, test_index in k_folds.split(storeC_data):
    storeC_data_train, storeC_data_test = storeC_data.iloc[train_index], storeC_data.iloc[test_index]

k_folds= KFold(n_splits= 10, shuffle= True, random_state= 0)
for train_index, test_index in k_folds.split(storeD_data):
    storeD_data_train, storeD_data_test = storeD_data.iloc[train_index], storeD_data.iloc[test_index]

k_folds= KFold(n_splits= 10, shuffle= True, random_state= 0)
for train_index, test_index in k_folds.split(storeE_data):
    storeE_data_train, storeE_data_test = storeE_data.iloc[train_index], storeE_data.iloc[test_index]

#divind the datasets into dependent and independent variables on the training data
y_storeA_data_train= storeA_data_train['Total Sales']
x_storeA_data_train= storeA_data_train.iloc[:, storeA_data_train.columns != 'Total Sales']
x_storeA_data_test= storeA_data_test.iloc[:, storeA_data_test.columns != 'Total Sales']
y_storeA_data_test= storeA_data_test['Total Sales']

y_storeB_data_train= storeB_data_train['Total Sales']
x_storeB_data_train= storeB_data_train.iloc[:, storeB_data_train.columns != 'Total Sales']
x_storeB_data_test= storeB_data_test.iloc[:, storeB_data_test.columns != 'Total Sales']
y_storeB_data_test= storeB_data_test['Total Sales']

y_storeC_data_train= storeC_data_train['Total Sales']
x_storeC_data_train= storeC_data_train.iloc[:, storeC_data_train.columns != 'Total Sales']
x_storeC_data_test= storeC_data_test.iloc[:, storeC_data_test.columns != 'Total Sales']
y_storeC_data_test= storeC_data_test['Total Sales']

y_storeD_data_train= storeD_data_train['Total Sales']
x_storeD_data_train= storeD_data_train.iloc[:, storeD_data_train.columns != 'Total Sales']
x_storeD_data_test= storeD_data_test.iloc[:, storeD_data_test.columns != 'Total Sales']
y_storeD_data_test= storeD_data_test['Total Sales']

y_storeA_data_train= storeA_data_train['Total Sales']
x_storeA_data_train= storeA_data_train.iloc[:, storeA_data_train.columns != 'Total Sales']
x_storeA_data_test= storeA_data_test.iloc[:, storeA_data_test.columns != 'Total Sales']
y_storeA_data_test= storeA_data_test['Total Sales']

y_storeE_data_train= storeE_data_train['Total Sales']
x_storeE_data_train= storeE_data_train.iloc[:, storeE_data_train.columns != 'Total Sales']
x_storeE_data_test= storeE_data_test.iloc[:, storeE_data_test.columns != 'Total Sales']
y_storeE_data_test= storeE_data_test['Total Sales']

# creating dependent and independent variables for all the data
y_storeA_data= storeA_data['Total Sales']
x_storeA_data= storeA_data.iloc[:, storeA_data.columns != 'Total Sales']

y_storeB_data= storeB_data['Total Sales']
x_storeB_data= storeB_data.iloc[:, storeB_data.columns != 'Total Sales']

y_storeC_data= storeC_data['Total Sales']
x_storeC_data= storeC_data.iloc[:, storeC_data.columns != 'Total Sales']

y_storeD_data= storeD_data['Total Sales']
x_storeD_data= storeD_data.iloc[:, storeD_data.columns != 'Total Sales']

y_storeE_data= storeE_data['Total Sales']
x_storeE_data= storeE_data.iloc[:, storeE_data.columns != 'Total Sales']


#performing linear regression on the dataset across each store

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt


#using linear regression on the store Data by cross validation predict function from sklearn
model= LinearRegression()
predictionA= cross_val_predict(model, x_storeA_data, y_storeA_data, cv=10)
predictionB= cross_val_predict(model, x_storeB_data, y_storeB_data, cv=10)
predictionC= cross_val_predict(model, x_storeC_data, y_storeC_data, cv=10)
predictionD= cross_val_predict(model, x_storeD_data, y_storeD_data, cv=10)
predictionE= cross_val_predict(model, x_storeE_data, y_storeE_data, cv=10)

#calculating cross_val_score for each store to compare 
scoreA_train= cross_val_score(model, x_storeA_data_train, y=y_storeA_data_train)
scoreA_test= cross_val_score(model, x_storeA_data_test, y=y_storeA_data_test)

scoreB_train= cross_val_score(model, x_storeB_data_train, y=y_storeB_data_train)
scoreB_test= cross_val_score(model, x_storeB_data_test, y=y_storeB_data_test)

scoreC_train= cross_val_score(model, x_storeC_data_train, y=y_storeC_data_train)
scoreC_test= cross_val_score(model, x_storeC_data_test, y=y_storeC_data_test)

scoreD_train= cross_val_score(model, x_storeD_data_train, y=y_storeD_data_train)
scoreD_test= cross_val_score(model, x_storeD_data_test, y=y_storeD_data_test)

scoreE_train= cross_val_score(model, x_storeE_data_train, y=y_storeE_data_train)
scoreE_test= cross_val_score(model, x_storeE_data_test, y=y_storeE_data_test)

scores_train=[scoreA_train, scoreB_train, scoreC_train, scoreD_train, scoreE_train]
scores_test= [scoreA_test, scoreB_test, scoreC_test, scoreD_test, scoreE_test]

#plotting the scores 
plt.title("Validation Curve")
plt.xlabel("stores")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2
plt.semilogx(np.mean(scores_train, axis=1), label="Training score",
             color="darkorange", lw=lw)

plt.semilogx( np.mean(scores_test, axis=1), label="Cross-validation score",
             color="navy", lw=lw)
plt.legend(loc="best")
plt.show()

#calculating the coefficient of determination for each model 
from sklearn.metrics import r2_score

regression_scoreA= r2_score(y_storeA_data, predictionA),
regression_scoreB= r2_score(y_storeB_data, predictionB),
regression_scoreC= r2_score(y_storeC_data, predictionC),
regression_scoreD= r2_score(y_storeD_data, predictionD),
regression_scoreE= r2_score(y_storeE_data, predictionE)

#Creating a function to plot the predicted and actual values of the linear model on each store
def plotlinearmodel(data, prediction):
    fig, ax = plt.subplots()
    ax.scatter(data, prediction, edgecolors=(0, 0, 0))
    ax.plot([data.min(), data.max()], [data.min(), data.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()
plotlinearmodel(y_storeC_data,predictionC )
plotlinearmodel(y_storeD_data,predictionD )
plotlinearmodel(y_storeE_data,predictionE )

#performing PCA to get top 3 principal features

from sklearn.decomposition import PCA
pca= PCA(n_components=3, svd_solver='full')
pca.fit(storeA_data)

pca.explained_variance_ratio_


#using random forest regressor to estimate the importance of features

from sklearn.ensemble import RandomForestRegressor
regressor= RandomForestRegressor(n_estimators=10, max_depth=10)
regressorA= regressor.fit(x_storeA_data, y_storeA_data)
regressorB= regressor.fit(x_storeB_data, y_storeB_data)
regressorC= regressor.fit(x_storeC_data, y_storeC_data)
regressorD= regressor.fit(x_storeD_data, y_storeD_data)
regressorE= regressor.fit(x_storeE_data, y_storeE_data)

#plot
def plottingfeatureimoprtance(data, clf):
    importances = pd.DataFrame({'feature':data.columns,'importance':np.round(clf.feature_importances_,3)})
    importances = importances.sort_values('importance',ascending=False).set_index('feature')
    print (importances)
    importances.plot.bar()

plottingfeatureimoprtance(x_storeA_data, regressorA)

plottingfeatureimoprtance(x_storeB_data, regressorB)
plottingfeatureimoprtance(x_storeC_data, regressorC)
plottingfeatureimoprtance(x_storeD_data, regressorD)
plottingfeatureimoprtance(x_storeE_data, regressorE)



