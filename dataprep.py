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

os.chdir(r'C:\Users\Kaja Amalie\Documents\Kaja\Kaggle\Kaggle')
df = pd.read_csv('train.csv')


#


#Create x and y:
X = df[['Pclass',
        'Sex',
        'Age',
        'SibSp',
        'Parch',
        'Embarked']]
y = df['Survived']




#Divide classes and make sure all values are numeric:
    
#1)Make classes into OneHotEncoder: 
tit_encoder = OneHotEncoder()
tit_encoder.fit(X[['Pclass']])
classes = tit_encoder.transform(X[['Pclass']]).todense()

X[['First_class', 'Second_class', 'Third_class']] = pd.DataFrame(classes)

#Delete values I no longer need: 
del X['Pclass']
del classes

# make male and female into 
tit_encoder = OneHotEncoder()
tit_encoder.fit(X[['Sex']])
Sexes = tit_encoder.transform(X[['Sex']]).todense()

X[['Female', 'Male']] = pd.DataFrame(Sexes)

del X['Sex']
del Sexes



#Make embarked into numerics
#Get rid of null values
titanic_imputer = SimpleImputer(strategy='most_frequent')
titanic_imputer.fit(X[['Embarked']])
X['Embarked'] = titanic_imputer.transform(X[['Embarked']])

tit_encoder = OneHotEncoder()
tit_encoder.fit(X[['Embarked']])
Embarked = tit_encoder.transform(X[['Embarked']]).todense()
X[['Cherbourg', 'Queenstown', 'Southampton']] = pd.DataFrame(Embarked)

del X['Embarked']
del Embarked



#Remove null values from age: 
titanic_imputer = SimpleImputer(strategy='mean')
titanic_imputer.fit(X[['Age']])
X['Age'] = titanic_imputer.transform(X[['Age']]) #Round to only have whole numbers

#Remove float numbers from age
X['Age'] = X['Age'].apply(np.floor)


#Create one column for fam. members:
X['Family_members'] = X['SibSp'] + X['Parch']
del X['SibSp']
del X['Parch']



# StandardScaler on 

data_std = StandardScaler()
data_std.fit(X[['Age','Family_members']])
data_std2 = data_std.transform(X[['Age','Family_members']])

X[['Age_sc', 'Family_members_sc']] = pd.DataFrame(data_std2, columns = ['Age_sc', 'Family_members_sc'])

del X['Family_members']
      



