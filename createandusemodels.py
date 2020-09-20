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

