# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null values using .isnull() function.
3. Import LabelEncoder and encode the dataset.
4. Import DecisionTreeRegressor from sklearn and apply the model on the dataset.
5. Predict the values of arrays.
6. Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.
7. Predict the values of array.
8. Apply to new unknown values.

## Program:
```py
# Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
# Developed by: GOKUL S
# RegisterNumber:  212222110011
```
```py
import pandas as pd
data=pd.read_csv("/content/Salary.csv")
data.head()
```
```py
data.info()
```
```py
data.isnull().sum()
```
```py
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['Position']=le.fit_transform(data['Position'])
data.head()
```
```py
x=data[['Position','Level']]
x
```
```py
y=data['Salary']
y
```
```py
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
```
```py
from sklearn.tree import DecisionTreeClassifier,plot_tree
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
```
```py
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse
```
```py
r2=metrics.r2_score(y_test,y_pred)
r2
```
```py
import matplotlib.pyplot as plt
dt.predict([[5,6]])
plt.figure(figsize=(20,8))
plot_tree(dt,feature_names=x.columns,filled=True)
plt.show()
```

## Output:
### Printing head

![image](https://github.com/SanjayRagavendar/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/91368803/f92ba820-3c03-4b0e-b2e7-cf6ecaac3e07)

### Printing info about dataset

![image](https://github.com/SanjayRagavendar/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/91368803/148ecec8-b241-4046-a2a3-c1b3161845b7)

### Counting the null values

![image](https://github.com/SanjayRagavendar/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/91368803/04c2aad3-d6e7-4263-a9c5-16542231360c)

### Label Encoding the Position Column with 

![image](https://github.com/SanjayRagavendar/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/91368803/38085260-8a91-428d-b174-153c8a102623)

### Spliting the dataset for dependent and independent values

![image](https://github.com/SanjayRagavendar/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/91368803/1904cc6f-fbcd-4838-aaff-9928cc2a2fd0)

![image](https://github.com/SanjayRagavendar/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/91368803/562cedde-bd9c-4111-8d3c-5451d01c1e2f)

### MSE for test_data

![image](https://github.com/SanjayRagavendar/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/91368803/8455b752-799f-4d21-8730-85023b9443bc)

### R2 value for test_data

![image](https://github.com/SanjayRagavendar/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/91368803/f60a65b5-26ff-4f94-9055-aee257573fbd)

### Printing Plot 

![image](https://github.com/SanjayRagavendar/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/91368803/04f40283-df79-4e23-95db-4c1b7f307322)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
