#%%
import pandas as pd 
import os
import cats.database as db

df = pd.read_csv("H:/Riset/Phyton Related/Data/dataset1/train1.csv")
df

#%%
soal1 = df[['annual_inc','revol_bal']]
soal1

#%%
import matplotlib.pyplot as plt
plt.scatter(df[['annual_inc']],df[['revol_bal']])
plt.show()

#%%
from sklearn import linear_model
from sklearn.metrics import r2_score
regr = linear_model.LinearRegression()
regr.fit(df[['annual_inc']],df[['revol_bal']])

regr_line = regr.predict(df[['annual_inc']])

plt.scatter(df[['annual_inc']],df[['revol_bal']])
plt.plot(df[['annual_inc']],regr_line)
plt.show()
print('Koefisien = ',regr.coef_)
print('R2 = ',r2_score(df[['revol_bal']],regr_line))
print(".\n"+db.get_random())

#%%
df.groupby('not_paid').count()
#%%
df.groupby('initial_list_status').count()
#%%
df.groupby('grdCtoA').count()

#%%
#Logistic Regression

 