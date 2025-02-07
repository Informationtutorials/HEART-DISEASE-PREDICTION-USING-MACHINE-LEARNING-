import numpy as np
import pandas as pd
import matplotlib.pyplot as plt from sklearn.metrics import confusion_matrix from sklearn.linear_model import LogisticRegression from sklearn.model_selection import train_test_split import seaborn
import os
for dirname, filenames in os.walk('/kaggle/input'):
for filename in filenames:
print(o
os.path.join(dimame, filename))
df=pd.read_csv('/kaggle/input/heart-disease-prediction-using-le
framingham.csv")
df.head(
df.shape
df.info()
df['education'].value_counts()
pd.crosstab(df['education'], df['Ten YearCHD']).plot(kind='bar', title="Education vs TenYearCHD")
df.drop(labels=['education'],axis=1,inplace=True) dfhead)
dlisna).sum().sort_values(ascending=False)

count_nulls-dfisna().sum(axis=1)

count_null_values=0

for i in count nulls:

if i>0: count_null_values+=1 print('Total number of rows with missing values is', count_null_values) print('since it is only'.round((count_null_values/len(dfindex))*100), 'percent of the entire dataset the rows with missing values are excluded.)

didropna(axis=0.inplace=True)
dfhead()
fig, axes = plt.subplots(2, 3. figsize=(15. 15)) Plot each crosstab in a different subplot pd.crosstab(df['male'], df[TenYearCHD].plot(kind='bar', ax=axes[0, 0], color=[#5DADE2", F580411, title="Male vs TenYearCHD") pd.crosstab(df currentSmoker), di['TenYearCHD']).plot(kind="bar', ax=axes[0, 1]
color=['*5DADE2', '#F5B041'], title="Current Smoker vs TenYearCHD")

pd.crosstab(df['BPMeds'], df['TenYearCHD']).plot(kind='bar', ax=axes[0, 2], color=['#5DADE2', '#F5B041'], title="BP Meds vs TenYearCHD") pd.crosstab(df['prevalentStroke'], df['TenYearCHD']).plot(kind='bar', ax=axes[1, 0], color=['#5DADE2', '#F5B041'], title="Prevalent Stroke vs TenYearCHD") pd.crosstab(df['prevalentHyp'], df['Ten YearCHD']).plot(kind='bar', ax=axes[1, 1], color=['#5DADE2', '#F58041'], title="Prevalent Hyp vs TenYearCHD") pd.crosstab(df['diabetes'], df['Ten YearCHD']).plot(kind='bar', ax=axes[1,2], color=['#5DADE2', '#F5B041'], title="Diabetes vs TenYearCHD")

df.groupby("TenYearCHD') [['age','cigsPerDay', 'totChol','sysBP', 'diaBP', 'BMI', 'heartRate','glucose']].mean()

X=df.drop([TenYearCHD'],axis=1)

X.head()
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix from sklearn.linear_model import LogisticRegression from sklearn.model_selection import train_test_split import seaborn
df.groupby('Ten YearCHD')

[['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']].mean()

X=df.drop(['TenYearCHD'],axis=1)

X.head()

y=df['TenYearCHD]

y

x_train,x_test,y_train.y_test=train_test_split(X,y.test_size=0.2) 

len(x_train)


len(x_test)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler)

x_train_scaled = scaler.fit_transform(x_train)

x_test_scaled = scaler.transform(x_test)

predict_heart_disease_model=LogisticRegression()
predict_heart_disease_model.fit(x_train_scaled,y_train)


y_predicted=predict_heart_disease_model.predict(x_test_scaled)

CM=confusion_matrix(y_test.y_predicted)

CM

plt.figure(figsize=(10,7))

sns.heatmap(CM,annot=True)

plt.xlabel('predict')

plt.ylabel('truth')

predict_heart_disease_model.score(x_test_scaled.y_test)
