#%%
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split

sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

#%%
data = pd.read_csv("banking.csv")
data = data.dropna()
data

#%%
sns.countplot(x='y',data=data, palette='hls')
plt.show()

#%%
sns.countplot(y='job',data=data)
plt.show()

#%%
data.drop(data.columns[[0, 3, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19]], axis=1, inplace=True)
data2 = pd.get_dummies(data, columns =['job', 'marital', 'default', 'housing', 'loan', 'poutcome'])
data2.drop(data2.columns[[12, 16, 18, 21, 24]], axis=1, inplace=True)
data2.columns

#%%
data2

#%%
sns.heatmap(data2.corr())
plt.show()

#%%
data2.corr()
#%%
#split data
x = data2.iloc[:,1:]
y = data2.iloc[:,0]

X_train, X_test, Y_train, Y_test = train_test_split(x,y,random_state = 0)

X_train.shape

#%%
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,Y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cf_matrix = confusion_matrix(Y_test,y_pred)
cf_matrix

#%%
#Report
from sklearn.metrics import classification_report
print(classification_report(Y_test,y_pred))











