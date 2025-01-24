# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 17:11:15 2025

@author: Jonathan Gonzalez

Machine Learning Regression Masterclass in Python 
By: Dr. Ryan Ahmed 
Platform: Udemy
Type: Compilation of videos

This project predicts customer interactions with Facebook ads using logistic regression. 
The dataset Facebook_Ads_2.csv is analyzed to explore relationships between features 
like time spent on site, salary, and click behavior. After visualizing key patterns, 
the data is preprocessed and split into training and testing sets. Model performance 
is evaluated using confusion matrices, classification metrics, and decision boundary 
visualizations, offering insights into customer behavior and ad targeting strategies.

Last Updated: 1/23/2024
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

trainingSet = pd.read_csv("Facebook_Ads_2.csv", encoding = "ISO-8859-1")

print(trainingSet.tail(5))

Click = trainingSet[ trainingSet["Clicked"]== 1 ]
noClick = trainingSet[ trainingSet["Clicked"]== 0 ]
print(Click)

print("Total = ", len(trainingSet))
print("Number of customers who clicked", len(Click))
print("Number of customers who did not click", len(noClick))

plt.close("all")
plt.figure()
sns.scatterplot(x = trainingSet["Time Spent on Site"], y = trainingSet["Salary"], hue = trainingSet["Clicked"])

plt.figure(figsize = (5,5))
sns.boxplot(x = "Clicked", y = "Salary", data = trainingSet, hue = trainingSet["Clicked"] )

plt.figure(figsize = (5,5))
sns.boxplot(x = "Clicked", y = "Time Spent on Site", data = trainingSet, hue = trainingSet["Clicked"] )

plt.figure()
trainingSet["Salary"].hist( bins = 40)

plt.figure()
trainingSet["Time Spent on Site"].hist( bins = 40)

trainingSet.drop(["Names", "emails", "Country"], axis = 1, inplace = True)

x = trainingSet.drop(["Clicked"], axis = 1 ).values
y = trainingSet["Clicked"].values

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x = sc.fit_transform(x)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)

classifier.fit(x_train, y_train)

y_predict = classifier.predict(x_test)

y_predict_train = classifier.predict(x_train)

from sklearn.metrics import classification_report, confusion_matrix

cm = confusion_matrix(y_train, y_predict_train)

sns.heatmap(cm, annot = True, fmt = "d")

plt.figure()
cm = confusion_matrix(y_test, y_predict)

sns.heatmap(cm, annot = True, fmt = "d")

print(classification_report(y_test, y_predict))

from matplotlib.colors import ListedColormap
x1, x2 = np.meshgrid(np.arange(start = x_train[: , 0].min() - 1, stop = x_train[:, 0].max() + 1 , step = 0.01 ) , np.arange(start = x_train[: , 1].min() - 1, stop = x_train[:, 1].max() + 1 , step = 0.01 ) )

plt.figure()
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T ).reshape(x1.shape), alpha = 0.65, cmap = ListedColormap(("green", "blue")) )
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_train)):
    plt.scatter(x_train[y_train == j, 0], x_train[y_train == j, 1], c = ListedColormap(("green","blue"))(i), label = j)
plt.title("Facebook Ad: Customer Click Prediction (Training set)", weight = "bold", size = 13)
plt.xlabel("Time Spent on Site")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()

x1, x2 = np.meshgrid(np.arange(start = x_test[: , 0].min() - 1, stop = x_test[:, 0].max() + 1 , step = 0.01 ) , np.arange(start = x_test[: , 1].min() - 1, stop = x_test[:, 1].max() + 1 , step = 0.01 ) )

plt.figure()
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T ).reshape(x1.shape), alpha = 0.65, cmap = ListedColormap(("green", "blue")) )
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_test)):
    plt.scatter(x_test[y_test == j, 0], x_test[y_test == j, 1], c = ListedColormap(("green","blue"))(i), label = j)
plt.title("Facebook Ad: Customer Click Prediction (Testing set)", weight = "bold", size = 13)
plt.xlabel("Time Spent on Site")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()