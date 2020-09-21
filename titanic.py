#Import lib
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsRegressor


os.chdir(r'C:\Users\Kaja Amalie\Documents\Kaja\Kaggle\Kaggle')
df = pd.read_csv('train.csv')

df_columns = df[['Pclass',
            'Sex',
            'Age',
            'SibSp',
            'Parch',
            'Embarked',
            'Survived']]

#Split the train data into train and test data
df_train, df_test = train_test_split(df_columns, test_size=0.2, random_state = 420)


#Create x and y:
X_train = df_train[['Pclass',
        'Sex',
        'Age',
        'SibSp',
        'Parch',
        'Embarked']]
y_train = df_train['Survived']


#Divide classes and make sure all values are numeric:

    
#1)Make classes into OneHotEncoder: 
tit_encoder = OneHotEncoder()
tit_encoder.fit(X_train[['Pclass']])
classes = tit_encoder.transform(X_train[['Pclass']]).todense()
X_train['First_class'] = classes[:,0]
X_train['Second_class'] = classes[:,1]
X_train['Third_class'] = classes[:,2]


#Delete values I no longer need: 
del X_train['Pclass']
del classes

# make male and female into 
tit_encoder = OneHotEncoder()
tit_encoder.fit(X_train[['Sex']])
Sexes = tit_encoder.transform(X_train[['Sex']]).todense()

X_train['Female'] = Sexes[:,0]
X_train['Male'] = Sexes[:,1]

del X_train['Sex']
del Sexes



#Make embarked into numerics
#Get rid of null values
titanic_imputer = SimpleImputer(strategy='most_frequent')
titanic_imputer.fit(X_train[['Embarked']])
X_train['Embarked'] = titanic_imputer.transform(X_train[['Embarked']])

tit_encoder = OneHotEncoder()
tit_encoder.fit(X_train[['Embarked']])
Embarked = tit_encoder.transform(X_train[['Embarked']]).todense()
X_train['Cherbourg'] = Embarked[:,0]
X_train['Queenstown'] =Embarked[:,1]
X_train['Southampton'] = Embarked[:,2]

del X_train['Embarked']
del Embarked



#Remove null values from age: 
titanic_imputer = SimpleImputer(strategy='mean')
titanic_imputer.fit(X_train[['Age']])
X_train['Age'] = titanic_imputer.transform(X_train[['Age']]) #Round to only have whole numbers

#Remove float numbers from age
X_train['Age'] = X_train['Age'].apply(np.floor)


#Create one column for fam. members:
X_train['Family_members'] = X_train['SibSp'] + X_train['Parch']
del X_train['SibSp']
del X_train['Parch']



# StandardScaler on 

data_std = StandardScaler()
data_std.fit(X_train[['Age','Family_members']])
data_std2 = data_std.transform(X_train[['Age','Family_members']])

X_train['Age_sc'] = data_std2[:,0]
X_train['Family_members_sc'] =  data_std2[:,1]


del X_train['Family_members']
del X_train['Age']



#########################################################################################
# Doing the same with the test data: 
X_test = df_test[['Pclass',
        'Sex',
        'Age',
        'SibSp',
        'Parch',
        'Embarked']]
y_test = df_test['Survived']


#Divide classes and make sure all values are numeric:

    
#1)Make classes into OneHotEncoder: 
tit_encoder = OneHotEncoder()
tit_encoder.fit(X_test[['Pclass']])
classes = tit_encoder.transform(X_test[['Pclass']]).todense()
X_test['First_class'] = classes[:,0]
X_test['Second_class'] = classes[:,1]
X_test['Third_class'] = classes[:,2]


#Delete values I no longer need: 
del X_test['Pclass']
del classes

# make male and female into 
tit_encoder = OneHotEncoder()
tit_encoder.fit(X_test[['Sex']])
Sexes = tit_encoder.transform(X_test[['Sex']]).todense()

X_test['Female'] = Sexes[:,0]
X_test['Male'] = Sexes[:,1]

del X_test['Sex']
del Sexes



#Make embarked into numerics
#Get rid of null values
titanic_imputer = SimpleImputer(strategy='most_frequent')
titanic_imputer.fit(X_test[['Embarked']])
X_test['Embarked'] = titanic_imputer.transform(X_test[['Embarked']])

tit_encoder = OneHotEncoder()
tit_encoder.fit(X_test[['Embarked']])
Embarked = tit_encoder.transform(X_test[['Embarked']]).todense()
X_test['Cherbourg'] = Embarked[:,0]
X_test['Queenstown'] =Embarked[:,1]
X_test['Southampton'] = Embarked[:,2]

del X_test['Embarked']
del Embarked



#Remove null values from age: 
titanic_imputer = SimpleImputer(strategy='mean')
titanic_imputer.fit(X_test[['Age']])
X_test['Age'] = titanic_imputer.transform(X_test[['Age']]) #Round to only have whole numbers

#Remove float numbers from age
X_test['Age'] = X_test['Age'].apply(np.floor)


#Create one column for fam. members:
X_test['Family_members'] = X_test['SibSp'] + X_test['Parch']
del X_test['SibSp']
del X_test['Parch']



# StandardScaler on 

data_std = StandardScaler()
data_std.fit(X_test[['Age','Family_members']])
data_std2 = data_std.transform(X_test[['Age','Family_members']])

X_test['Age_sc'] = data_std2[:,0]
X_test['Family_members_sc'] =  data_std2[:,1]


del X_test['Family_members']
del X_test['Age']