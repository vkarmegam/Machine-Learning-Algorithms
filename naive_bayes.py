import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

df = read_csv(r'd:\Karmegam\pgm\ML\Naive-Bayes-Classification-Data.csv',low_memory=False)
x=df.drop('diabetes',axis=1)
y=df['diabetes']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)
model=GaussianNB()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(y_pred)
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)
