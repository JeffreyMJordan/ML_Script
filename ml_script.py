import pandas as pd 
import numpy as np
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
j2017 = "jan2017.csv"
# f2017 = "/Users/christine/Desktop/python/dffeb2017.csv"
print(j2017)
db = pd.read_csv(j2017, low_memory=False)
# de = pd.read_csv(f2017, low_memory=False)

#Corresponds to 'ArrDelayMintues
a = db['ARR_DELAY_NEW']



a = a.fillna(0).astype(int)
idx = 0
for val in np.nditer(a):
	if val <= 15 and val>0:
		a[idx] = 15
	if val<=30 and val>15:
		a[idx] = 30
	if val<=45 and val>30:
		a[idx] = 45
	if val>45:
		a[idx] = 46
	idx = idx+1


keep_col = ['MONTH', 'AIRLINE_ID', 'ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID', 'DISTANCE']
d = db[keep_col]
ab = pd.concat([d,a], axis=1)
ab['DISTANCE'] = ab['DISTANCE'].astype(int)
ab['ARR_DELAY_NEW'] = ab['ARR_DELAY_NEW'].astype(int)

array = ab.values     
X = array[:,0:5]
Y = array[:,5] 
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC()))
results = []
names = ['MONTH', 'AIRLINE_ID', 'ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID', 'DISTANCE', 'ARR_DELAY_NEW']
seed = 7
scoring = 'accuracy'

for name, model in models:
		kfold = model_selection.KFold(n_splits=10, random_state=seed)
		cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
		results.append(cv_results)
		names.append(name)
		msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
		print(msg)



lr = LogisticRegression() 
lr.fit(X_train, Y_train) 
predictions = lr.predict_proba(X_validation)
print(lr.classes_)
print(predictions)

cart = DecisionTreeClassifier()
cart.fit(X_train, Y_train)
predictions = cart.predict_proba(X_validation)
print(predictions)
# print(accuracy_score(Y_validation, predictions))
# print(confusion_matrix(Y_validation, predictions))
# print(classification_report(Y_validation, predictions))
