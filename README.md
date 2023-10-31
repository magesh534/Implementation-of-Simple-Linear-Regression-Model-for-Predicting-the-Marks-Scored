# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas

## Program:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/student_scores.csv')
#displaying the content in datafile
df.head()

df.tail()

x=df.iloc[:,:-1].values
x

y=df.iloc[:,1].values
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
y_pred
y_test

y_pred

#graph plot for training data
plt.scatter(x_train,y_train,color="darkseagreen")
plt.plot(x_train,regressor.predict(x_train),color="plum")
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#graph plot for test data
plt.scatter(x_test,y_test,color="darkblue")
plt.plot(x_test,regressor.predict(x_test),color="plum")
plt.title("Hours vs Scores (Test set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```

## Output:
1. df.head()
![ml 1](https://github.com/magesh534/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/135577936/24458f64-35f9-4347-832c-cbafe6523716)
2. df.tail()
![ml 2](https://github.com/magesh534/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/135577936/ade86d1c-5c11-45b4-9e26-e8a001c33331)
3.Array value of X
![ml 3](https://github.com/magesh534/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/135577936/b7ac3bd4-1260-4959-ab68-3cc22ea7eba4)
4.Array value of Y
![ml 4](https://github.com/magesh534/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/135577936/bb6304e9-4ef3-4df1-99ca-98f6ec1de3e2)
5. Values of Y prediction
![ml 5](https://github.com/magesh534/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/135577936/39bae5bf-c00e-4c9e-ac98-65f1eb82cc24)
6.Array values of Y test
![ml 6](https://github.com/magesh534/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/135577936/25592a50-268a-440d-9165-3988437a1b98)
7.Training Set Graph
![ml 7](https://github.com/magesh534/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/135577936/2b387912-598d-4de4-9dc9-ec6196be95c2)
8.Test Set Graph
![ml8](https://github.com/magesh534/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/135577936/26c88334-e395-4d9e-888d-392f72fac961)
9.Values of MSE, MAE and RMSE
![ml 9](https://github.com/magesh534/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/135577936/b31bb641-bd7a-4ac5-ac35-770f29234c77)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
