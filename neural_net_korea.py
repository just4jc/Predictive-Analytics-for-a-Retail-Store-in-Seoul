# -*- coding: utf-8 -*-
"""
Created on Thu May 10 00:55:46 2018

@author: ojhadiwesh
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#import the training set
storeA= pd.read_csv('D:\PA\storeA.csv')
storeB= pd.read_csv('D:\PA\storeB.csv')
storeC= pd.read_csv('D:\PA\storeC.csv')
storeD= pd.read_csv('D:\PA\storeD.csv')
storeE= pd.read_csv('D:\PA\storeE.csv')
storeA= storeA.iloc[:, storeA.columns!='Date']
storeB= storeB.iloc[:, storeB.columns!='Date']
storeC= storeC.iloc[:, storeC.columns!='Date']
storeD= storeD.iloc[:, storeD.columns!='Date']
storeE= storeE.iloc[:, storeE.columns!='Date']
storeA= storeA.iloc[:, storeA.columns!='Store Name']
storeB= storeB.iloc[:, storeB.columns!='Store Name']
storeC= storeC.iloc[:, storeC.columns!='Store Name']
storeD= storeD.iloc[:, storeD.columns!='Store Name']
storeE= storeE.iloc[:, storeE.columns!='Store Name']
storeA= storeA.iloc[:, storeA.columns!='Code']
storeB= storeB.iloc[:, storeB.columns!='Code']
storeC= storeC.iloc[:, storeC.columns!='Code']
storeD= storeD.iloc[:, storeD.columns!='Code']
storeE= storeE.iloc[:, storeE.columns!='Code']
def filloutlook(data):
    outlook= data['Outlook']
    outlook= outlook.fillna('Sunny')
    data['Outlook']= outlook
filloutlook(storeA)
filloutlook(storeB)
filloutlook(storeC)
filloutlook(storeD)
filloutlook(storeE)

#Imputing the japanese tourists data by filling the mean of the tourist data
def touristimpute(data):
    data['Japanese Tourists']= data['Japanese Tourists'].fillna(data['Japanese Tourists'].mean())

touristimpute(storeA)
touristimpute(storeB)
touristimpute(storeC)
touristimpute(storeD)
touristimpute(storeE)
    
#labelling the data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
def labelling(data):
    labelencoder_data = LabelEncoder()
    data['Weekday'] = labelencoder_data.fit_transform(data['Weekday'])
    data['Outlook'] = labelencoder_data.fit_transform(data['Outlook'])
labelling(storeA)
labelling(storeB)
labelling(storeC)
labelling(storeD)
labelling(storeE)

#test train data

from sklearn.model_selection import KFold

k_folds= KFold(n_splits= 10, shuffle= True, random_state= 0)
for train_index, test_index in k_folds.split(storeA):
    storeA_train, storeA_test = storeA.iloc[train_index], storeA.iloc[test_index]

k_folds= KFold(n_splits= 10, shuffle= True, random_state= 0)
for train_index, test_index in k_folds.split(storeB):
    storeB_train, storeB_test = storeB.iloc[train_index], storeB.iloc[test_index]

k_folds= KFold(n_splits= 10, shuffle= True, random_state= 0)
for train_index, test_index in k_folds.split(storeC):
    storeC_train, storeC_test = storeC.iloc[train_index], storeC.iloc[test_index]

k_folds= KFold(n_splits= 10, shuffle= True, random_state= 0)
for train_index, test_index in k_folds.split(storeD):
    storeD_train, storeD_test = storeD.iloc[train_index], storeD.iloc[test_index]

k_folds= KFold(n_splits= 10, shuffle= True, random_state= 0)
for train_index, test_index in k_folds.split(storeE):
    storeE_train, storeE_test = storeE.iloc[train_index], storeE.iloc[test_index]

#encode the data

from sklearn.preprocessing import MinMaxScaler
sc= MinMaxScaler(feature_range=(0,1))
storeA_train_scaled= sc.fit_transform(storeA_train)
storeB_train_scaled= sc.fit_transform(storeB_train)
storeC_train_scaled= sc.fit_transform(storeC_train)
storeD_train_scaled= sc.fit_transform(storeD_train)
storeE_train_scaled= sc.fit_transform(storeE_train)

#creating a data structure with 60 timesteps and one output
X_trainA=[]
y_trainA=[]
for i in range(10, 422):
    X_trainA.append(storeA_train_scaled[i-10:i,np.r_[2,3,15,16, 19]])
    y_trainA.append(storeA_train_scaled[i,np.r_[2,3,15,16, 19]])
    
X_trainA,y_trainA=np.array(X_trainA), np.array(y_trainA)

X_trainB=[]
y_trainB=[]
for i in range(10, 507):
    X_trainB.append(storeB_train_scaled[i-10:i,np.r_[2, 19]])
    y_trainB.append(storeB_train_scaled[i,np.r_[2, 19]])
    
X_trainB,y_trainB=np.array(X_trainB), np.array(y_trainB)

X_trainC=[]
y_trainC=[]
for i in range(10, 354):
    X_trainC.append(storeC_train_scaled[i-10:i,np.r_[2,3,15,16, 19]])
    y_trainC.append(storeC_train_scaled[i,np.r_[2,3,15,16, 19]])
    
X_trainC,y_trainC=np.array(X_trainC), np.array(y_trainC)

X_trainD=[]
y_trainD=[]
for i in range(10, 504):
    X_trainD.append(storeD_train_scaled[i-10:i,np.r_[2, 19]])
    y_trainD.append(storeD_train_scaled[i,np.r_[2, 19]])
    
X_trainD,y_trainD=np.array(X_trainD), np.array(y_trainD)

X_trainE=[]
y_trainE=[]
for i in range(5, 423):
    X_trainE.append(storeE_train_scaled[i-5:i,0])
    y_trainE.append(storeE_train_scaled[i,0])
    
X_trainE,y_trainE=np.array(X_trainE), np.array(y_trainE)

#reshape
X_trainA=np.reshape(X_trainA, (X_trainA.shape[0], X_trainA.shape[1], 5))
X_trainB=np.reshape(X_trainB, (X_trainB.shape[0], X_trainB.shape[1], 2))
X_trainC=np.reshape(X_trainC, (X_trainC.shape[0], X_trainC.shape[1], 5))
X_trainD=np.reshape(X_trainD, (X_trainD.shape[0], X_trainD.shape[1], 2))
X_trainE=np.reshape(X_trainE, (X_trainE.shape[0], X_trainE.shape[1], 1))


#start an RNN for stpre A
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import Sequential


#intialisse the regresson
regressor= Sequential()

#the first layer of the LSTM and droput
regressor.add(LSTM(units=50, return_sequences= True, input_shape=( X_trainC.shape[1], 5)))
regressor.add(Dropout(0.2))


#second LSTM layer
regressor.add(LSTM(units=50, return_sequences= True))
regressor.add(Dropout(0.2))

#third LSTM layer
regressor.add(LSTM(units=50, return_sequences= True))
regressor.add(Dropout(0.2))

#fourth LSTM layer
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))


#the output layer
regressor.add(Dense(units=5))

#compiling the regressor
regressor.compile(optimizer='adam', loss='mean_squared_error', metrics= ['acc'])

#fit the model on the train data
regressor.fit(X_trainC, y_trainC, batch_size=40, epochs=100)


#making predictions 
dataset_total= pd.concat((storeC_train.iloc[:, np.r_[2,3,15,16, 19]], storeC_test.iloc[:, np.r_[2,3,15,16, 19]]), axis=0)
inputs= dataset_total[len(dataset_total)-len(storeC_test)-10:].values
inputs=sc.fit_transform(inputs)
X_test=[]
for i in range(10, len(storeC_test)+10):
    X_test.append(inputs[i-10:i,0:5])
X_test=np.array(X_test)

X_test=np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 5))

predicted= regressor.predict(X_test)

predicted=sc.inverse_transform(predicted)
predicted_total_sales= predicted[:,0]

real_total_sales= storeC_test.iloc[: ,2:3].values

#visualize the prediction against the real price
plt.plot(real_total_sales, color='red', label='Real Total sales for store C')

plt.plot(predicted_total_sales, color='blue', label='Predicted Total Sales for Store C')

plt.title('Total Sales prediction')
plt.xlabel('Days')
plt.ylabel('Total Sales in $')
plt.legend()
plt.show()