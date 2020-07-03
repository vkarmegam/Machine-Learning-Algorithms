import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from scipy import stats
import datetime
#%matplotlib inline

dataset=pd.read_csv('D:\Karmegam\pgm\ML\Weather.csv')

#5
'''
print()
print(dataset.head)
#dt
print(dataset.dtypes)
#null
print('Null is --> ', dataset.isnull().sum())
print()
#sum
print(dataset.describe())'''

print('SHAPE...\n Data has {} rows and {} Columns...'.format(dataset.shape[0],dataset.shape[1]))

dataset.plot(x='MinTemp',y='MaxTemp',style='o')
plt.title('Karmegam - MinTemp vs MaxTemp')
plt.xlabel('MinTemp')
plt.ylabel('MaxTemp')
plt.show()

print('fig is ', plt.figure(figsize=(20,15)))
print('tight ', plt.tight_layout())

plt.figure(figsize=(15,10))
plt.tight_layout()
sns.distplot(dataset['MaxTemp'],  kde=True, rug=False)
plt.show()

x=dataset['MinTemp'].values.reshape(-1,1)
y=dataset['MaxTemp'].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train) #training the algorithm

#To retrieve the intercept:
print(regressor.intercept_)
#For retrieving the slope:
print(regressor.coef_)

y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df1 = df.head(25)
#.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))