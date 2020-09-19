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


#Create x and y:
X = df[['Pclass',
        'Sex',
        'Age',
        'SibSp',
        'Parch',
        'Embarked']]
y = df['Survived']

#Get rid of null values

titanic_imputer = SimpleImputer(strategy='mean')
titanic_imputer.fit(X)



np_train_features = time_imputer.transform(X)
features_column_names = list(X)

X_filled = pd.DataFrame(np_train_features, columns=features_column_names)
#Take this one further if you have missing values - I'm not using this one further 
#now because I know that I have no missing values in X. c. 
#make them numeric
age = cat_encoder.transform(df_use[['age_text']]).todense()