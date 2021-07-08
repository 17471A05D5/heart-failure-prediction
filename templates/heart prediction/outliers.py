import numpy as np #used for arrays
import pandas as pd #for analyzing the data
#import seaborn as sns;sns.set()
from sklearn.model_selection import train_test_split #to split the data into test and training 
from sklearn.preprocessing import StandardScaler  
#import matplotlib.pyplot as plt
df=pd.read_csv(".\heart_failure_clinical_records_dataset.csv")
df.head()
df.dtypes
df.info()
df.shape
categorical=['sex','anaemia','high_blood_pressure','smoking','DEATH_EVENT']
numerical=['age','creatinine_phosphokinase', 'diabetes', 'ejection_fraction','platelets', 'serum_creatinine', 'serum_sodium','time']
cols=['sex','anaemia','high_blood_pressure','smoking','age','creatinine_phosphokinase', 'diabetes', 'ejection_fraction','platelets'
      , 'serum_creatinine', 'serum_sodium','time','DEATH_EVENT']
for i in cols:
    df[i]=df[i].fillna(df[i].dropna().mode()[0])
df.info()
#splitting dataset into training and test data
y=df.DEATH_EVENT
x=df.drop('DEATH_EVENT',axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
st_x= StandardScaler()  
x_train= st_x.fit_transform(x_train)  
x_test= st_x.transform(x_test)
print(x_train)  
#Logistic regression algorithm
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
trained_model=classifier.fit(x_train,y_train)
trained_model.fit(x_train,y_train )
# Predicting the Test set results
y_pred = classifier.predict(x_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
cm1 = confusion_matrix(y_test, y_pred)
print(cm1)
#print("Accuracy score of train LogisticRegression")
#print(accuracy_score(y_train, trained_model.predict(x_train))*100)
print("Accuracy score of test LogisticRegression")
a1=accuracy_score(y_test, y_pred)*100
print(a1)
#decision tree algorithm
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(random_state = 0)
trained_model=classifier.fit(x_train,y_train)
trained_model.fit(x_train,y_train )
# Predicting the Test set results
y_pred = classifier.predict(x_test)
# Making the Confusion Matrix
cm2= confusion_matrix(y_test, y_pred)
print(cm2)
#print("Accuracy score of train Decision tree")
#print(accuracy_score(y_train, trained_model.predict(x_train))*100)
print("Accuracy score of test Decision tree")
a2=accuracy_score(y_test, y_pred)*100
print(a2)

#RandomForestClassifier algorithm
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(random_state = 0)
trained_model=classifier.fit(x_train,y_train)
trained_model.fit(x_train,y_train )
# Predicting the Test set results
y_pred = classifier.predict(x_test)
# Making the Confusion Matrix
cm3 = confusion_matrix(y_test, y_pred)
print(cm3)
#print("Accuracy score of train RandomForestClassifier")
#print(accuracy_score(y_train, trained_model.predict(x_train))*100)
print("Accuracy score of test RandomForestClassifier")
a3=accuracy_score(y_test, y_pred)*100
print(a3)
#KNeighborsClassifier algorithm
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()
trained_model=classifier.fit(x_train,y_train)
trained_model.fit(x_train,y_train )
# Predicting the Test set results
y_pred = classifier.predict(x_test)
# Making the Confusion Matrix
cm4 = confusion_matrix(y_test, y_pred)
print(cm4)
#print("Accuracy score of train KNeighborsClassifier")
#print(accuracy_score(y_train, trained_model.predict(x_train))*100)
print("Accuracy score of test KNeighborsClassifier")
a4=accuracy_score(y_test, y_pred)*100
print(a4)
#support vector machine algorithm
from sklearn.svm import SVC
classifier = SVC()
trained_model=classifier.fit(x_train,y_train)
trained_model.fit(x_train,y_train )
# Predicting the Test set results
y_pred = classifier.predict(x_test)
# Making the Confusion Matrix
cm5 = confusion_matrix(y_test, y_pred)
print(cm5)
#print("Accuracy score of train support vector machine(svm)")
#print(accuracy_score(y_train, trained_model.predict(x_train))*100)
print("Accuracy score of test support vector machine(svm)")
a5=accuracy_score(y_test, y_pred)*100
print(a5)
#naive bayes algorithm
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
trained_model=classifier.fit(x_train,y_train)
trained_model.fit(x_train,y_train )
# Predicting the Test set results
y_pred = classifier.predict(x_test)
# Making the Confusion Matrix
cm6 = confusion_matrix(y_test, y_pred)
print(cm6)
#print("Accuracy score of train Naive bayes")
#print(accuracy_score(y_train, trained_model.predict(x_train))*100)
print("Accuracy score of test Naive bayes")
a6=accuracy_score(y_test, y_pred)*100
print(a6)
