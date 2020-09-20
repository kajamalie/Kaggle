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
   
#create train before staring with the test data   
X_train = X
y_train = y

del X
del y


# Doing the same with the test data: 
os.chdir(r'C:\Users\Kaja Amalie\Documents\Kaja\Kaggle\Kaggle')
df2 = pd.read_csv('test.csv')

del X_train['Age']


# Doing exactly the same for test data:
#Create x and y:
X_test = df[['Pclass',
        'Sex',
        'Age',
        'SibSp',
        'Parch',
        'Embarked']]



#Divide classes and make sure all values are numeric:
    
#1)Make classes into OneHotEncoder: 
tit_encoder = OneHotEncoder()
tit_encoder.fit(X_test[['Pclass']])
classes = tit_encoder.transform(X_test[['Pclass']]).todense()

X_test[['First_class', 'Second_class', 'Third_class']] = pd.DataFrame(classes)

#Delete values I no longer need: 
del X_test['Pclass']
del classes

# make male and female into 
tit_encoder = OneHotEncoder()
tit_encoder.fit(X_test[['Sex']])
Sexes = tit_encoder.transform(X_test[['Sex']]).todense()

X_test[['Female', 'Male']] = pd.DataFrame(Sexes)

del X_test['Sex']
del Sexes



#Make embarked into numerics
#Get rid of null values
titanic_imputer2 = SimpleImputer(strategy='most_frequent')
titanic_imputer2.fit(X_test[['Embarked']])
X_test['Embarked'] = titanic_imputer2.transform(X_test[['Embarked']])

tit_encoder2 = OneHotEncoder()
tit_encoder2.fit(X_test[['Embarked']])
Embarked = tit_encoder2.transform(X_test[['Embarked']]).todense()
X_test[['Cherbourg', 'Queenstown', 'Southampton']] = pd.DataFrame(Embarked)

del X_test['Embarked']
del Embarked



#Remove null values from age: 
titanic_imputer3 = SimpleImputer(strategy='mean')
titanic_imputer3.fit(X_test[['Age']])
X_test['Age'] = titanic_imputer3.transform(X_test[['Age']]) #Round to only have whole numbers

#Remove float numbers from age
X_test['Age'] = X_test['Age'].apply(np.floor)


#Create one column for fam. members:
X_test['Family_members'] = X_test['SibSp'] + X_test['Parch']
del X_test['SibSp']
del X_test['Parch']

#Remove null values from age: 
titanic_imputer = SimpleImputer(strategy='mean')
titanic_imputer.fit(X[['Age']])
X['Age'] = titanic_imputer.transform(X[['Age']]) #Round to only have whole numbers

#Remove float numbers from age
X['Age'] = X['Age'].apply(np.floor)


# StandardScaler on 

data_std_t = StandardScaler()
data_std_t.fit(X_test[['Age','Family_members']])
data_std2_t = data_std_t.transform(X_test[['Age','Family_members']])

X_test[['Age_sc', 'Family_members_sc']] = pd.DataFrame(data_std2_t, columns = ['Age_sc', 'Family_members_sc'])

del X_test['Family_members']
   

del X_test['Age']
