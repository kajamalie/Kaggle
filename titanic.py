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

##############################################################################################
TESTS


#Create and Train a linearregression algorithm with the training data. 

lin_model = LinearRegression()
lin_model.fit(X=X_train, y=y_train)


#Make prediction using the training and test data
#TRAIN DATA
y_train_pred = lin_model.predict(X_train)
mean_absolute_error(y_train, y_train_pred) # 0.2976086979535834
np.sqrt(mean_squared_error(y_train, y_train_pred)) #  0.38201420934553143

#Predict with a nain model(using mean as prediction every time)
y_train_mean = np.mean(y_train)*np.ones(y_train.shape)
mean_absolute_error(y_train, y_train_mean) #0.46944830198207294
np.sqrt(mean_squared_error(y_train_mean, y_train)) #0.4844833856707953

########################
#TEST DATA 
y_test_pred = lin_model.predict(X_test)
mean_absolute_error(y_test, y_test_pred) #0.275562169097866
np.sqrt(mean_squared_error(y_test, y_test_pred)) #0.36014012767514536

y_test_mean = np.mean(y_test)*np.ones(y_test.shape)
mean_absolute_error(y_test, y_test_mean) #0.48500358915139985
np.sqrt(mean_squared_error(y_test_mean, y_test)) #0.49244471220198915

##################################################################################################################3
# Logistic regression model

model_lr = LogisticRegression(max_iter = 1000)
model_lr.fit(X=X_train, y=y_train)



# Make prediction using the training and test data

#TRAIN PRED
y_train_pred2 = model_lr.predict(X_train)

#TEST PRED
y_test_pred2 = model_lr.predict(X_test)

#Naive models
y_train_pred2_naive =np.zeros(X_train.shape)
y_test_pred2_naive = np.zeros(y_test.shape)


# Accuracy_score 
train_acc = accuracy_score(y_train, y_train_pred2) #0.800561797752809

test_acc= accuracy_score(y_test, y_test_pred2 ) # 0.8324022346368715

test_acc_naive = accuracy_score(y_test, y_test_pred2_naive) # 0.5865921787709497

#MATRIX:

matrix_train = confusion_matrix(y_train, y_train_pred2) # [388, 56], [86, 182]

matrix_test = confusion_matrix(y_test, y_test_pred2) # [95, 10], [20, 54]
matrix_test_naive = confusion_matrix(y_test, y_test_pred2_naive) # [105, 0], [74, 0]


precision_train =  182/(182+56) # 0.7647058823529411
recall_train = 182/(182+86) # 0.6791044776119403

precision_test =  54/(54+10) #  0.84375
recall_test = 54/(54+20) # 0.7297297297297297




########################################################################################


#KN model: 
    
Knear_mod = KNeighborsRegressor(n_neighbors=10)
Knear_mod.fit(X=X_train, y=y_train)

knear_y_train_pred = Knear_mod.predict(X_train)
knear_y_test_pred = Knear_mod. predict(X_test)

print(mean_absolute_error(y_train ,knear_y_train_pred)) #0.2279494382022472
print(np.sqrt(mean_squared_error(y_train,knear_y_train_pred))) #0.3403773517333868
print(mean_absolute_error(y_test, knear_y_test_pred)) # 0.24357541899441346
print(np.sqrt(mean_squared_error(y_test, knear_y_test_pred))) # 0.3534348612082832
  

####################################################################################
#SUPPORT VECTOR MACHINES (SVM)
from sklearn.svm import SVC, SVR

SVC_model = SVC(kernel = 'poly', degree=3, coef0=1, C=10)
SVC_model.fit(X=X_train, y=y_train)

#TRAIN PRED
y_train_pred_SVC = SVC_model.predict(X_train)

#TEST PRED
y_test_pred_SVC = SVC_model.predict(X_test)
train_acc = accuracy_score(y_train, y_train_pred_SVC) #C5 : 0.8300561797752809 #C1 : 0.8342696629213483 #C10 : 0.8286516853932584
test_acc= accuracy_score(y_test, y_test_pred_SVC) # C5 : 0.8100558659217877 # C1: 0.8212290502793296 #C10 : 0.8100558659217877


matrix_test = confusion_matrix(y_test, y_test_pred_SVC) #[101, 4], [28, 46] #C10 [98, 7], [27, 47]
matrix_test2 = confusion_matrix(y_train, y_train_pred_SVC) #[426, 18], [100, 168] #C10 [420, 24], [98, 170]

#Precision: TP/ (TP+FP) (TRAIN)
#1
168/(168+18)  #= 90
#Recall: TP/ (TP+FN)
168/(168+100) #= 62.7

#10 
#pr:
170/(170 + 24) #0.8762886597938144
#r
170/(170 + 98) #0.6343283582089553
    

#Precision: TP/ (TP+FP) (TEST)
46/(46+4)  #= 92
#Recall: TP/ (TP+FN)
46/(46+28) #= 62.7

#10 
#pr:
47/(47+7) #0.8703703703703703
#r
47/(47+27) #0.6351351351351351
