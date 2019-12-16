#import libabry
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import dataset
dataset = pd.read_csv('Data_bank_churn.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values
# convert categories data (geography and gender)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder= LabelEncoder()
X[:, 1] = labelencoder.fit_transform(X[:, 1])
X[:, 2] = labelencoder.fit_transform(X[:, 2])
X_df = pd.DataFrame(X)

#dummy's varible of geogrphy's data
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

# delete 0ne dummy data on geography column
X = X[:, 1:]

# split data to data set and data test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# import keras library
import keras
from keras.models import Sequential
from keras.layers import Dense

#feature scaling process
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#ANN initialization
classificationEngine = Sequential()
 
# Add input layer and first hidden layer
classificationEngine.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
 
# add second hidden layer
classificationEngine.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
 
# add output layer
classificationEngine.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
 
# started ann
classificationEngine.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
 
# Fitting ANN ke training set
classificationEngine.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)
 
# predict result test
y_pred = classificationEngine.predict(X_test)
y_pred = (y_pred > 0.5)
 
#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
 

