# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 14:42:54 2021

@author: HEMA
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pickle
df=pd.read_csv(".\heart_failure_clinical_records_dataset.csv")
categorical=['sex','anaemia','high_blood_pressure','smoking','DEATH_EVENT']
numerical=['age','creatinine_phosphokinase', 'diabetes', 'ejection_fraction','platelets', 'serum_creatinine', 'serum_sodium','time']
cols=['sex','anaemia','high_blood_pressure','smoking','age','creatinine_phosphokinase', 'diabetes', 'ejection_fraction','platelets', 'serum_creatinine', 'serum_sodium','time','DEATH_EVENT']
for i in cols:
    df[i]=df[i].fillna(df[i].dropna().mode()[0])
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for i in categorical:
    df[i] = le.fit_transform(df[i])

    
train=df.iloc[:,0:7]
test=df.iloc[:,-1]
from sklearn.feature_selection import RFE
y=df.DEATH_EVENT
x=df.drop('DEATH_EVENT',axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
st_x= StandardScaler()  
x_train= st_x.fit_transform(x_train)  
x_test= st_x.transform(x_test)

RF=RandomForestClassifier()
RF.fit(x_train,y_train)

pickle.dump(RF, open('model1.pkl','wb'))
model = pickle.load(open('model1.pkl','rb'))
