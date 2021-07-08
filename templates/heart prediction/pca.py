import numpy as np
import pandas as pd
import seaborn as sns;sns.set()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  
import matplotlib.pyplot as plt
df=pd.read_csv('.\heart_failure_clinical_records_dataset.csv')
sns.heatmap(df.isnull(),cbar=False)
df.head()
df.dtypes
df.info()
df.shape
categorical=['sex','anaemia','high_blood_pressure','smoking','DEATH_EVENT']
numerical=['age','creatinine_phosphokinase', 'diabetes', 'ejection_fraction','platelets', 'serum_creatinine', 'serum_sodium','time']
cols=['sex','anaemia','high_blood_pressure','smoking','age','creatinine_phosphokinase', 'diabetes', 'ejection_fraction','platelets'
      , 'serum_creatinine', 'serum_sodium','time','DEATH_EVENT']


#filling missing values with mean
for i in numerical:
    df[i].fillna(df[i].mean(),inplace=True)
    df.info()
#filling missing values with mode

for i in cols:
    df[i]=df[i].fillna(df[i].dropna().mode()[0])
df.info()
#changing categorical variables into numerical variables
cleanup_nums = {"sex":    {"male": 0, "female": 1},
                "anaemia":    {"yes": 0, "no": 1},
                "high_blood_pressure":     {"yes": 0, "no": 1},
                "smoking":    {"yes": 0, "no": 1},
                
                "DEATH_EVENT":{"Die": 0, "notDie": 1}}
df.replace(cleanup_nums, inplace=True)
df.dtypes
df.shape
sns.heatmap(df.isnull(),cmap='coolwarm',cbar=False)
corrmat=df.corr()
plt.figure(figsize=(20,20))
heat_map=sns.heatmap(corrmat,annot=True,cmap='YlGnBu', vmin=None, vmax=None,linewidths=0)
plt.show()
# Select upper triangle of correlation matrix
corr_matrix = df.corr().abs()
upper = np.triu(np.ones_like(corr_matrix,dtype=bool))
k=corr_matrix.mask(upper)
# Find index of feature columns with correlation greater than 0.4
to_drop = [column for column in k.columns if any(k[column] > 0.7)]
# Drop features
after_dropped=df.drop(df[to_drop], axis=1)
print(to_drop)
print(len(after_dropped.columns))

X = after_dropped.iloc[:, :-1].values # attributes to determine dependent variable / Class
y = after_dropped.iloc[:, -1].values# dependent variable / Class
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=42)
X1_train, X1_test, y1_train, y1_test = train_test_split(X, y, test_size=0.20, random_state=42)
plt.figure(figsize = (19,19))
sns.heatmap(after_dropped.corr(), annot = True, cmap = 'coolwarm') # looking for strong correlations with "class" row
plt.title("Correlation Heatmap after Removing columns", fontsize=12)
plt.show()

#splitting dataset into training and test data
y=df.classification
x=df.drop('classification',axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
st_x= StandardScaler()  
x_train= st_x.fit_transform(x_train)  
x_test= st_x.transform(x_test)
print(x_train)

#pca
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
PX_train=pca.fit_transform(x_train)
PX_test=pca.transform(x_test)
