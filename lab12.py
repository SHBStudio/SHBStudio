
# Lab Title: To implement simple Linear Regression and Plot the graph
# Objective: The objective of this lab is to implement simple linear regression in Python
#            and visualize the linear regression line on a scatter plot. 
#            Simple linear regression is a method to model the relationship 
#            between a single independent variable and a dependent variable. 
#            This lab aims to provide hands-on experience in understanding and implementing basic linear regression.

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston

# Load the Boston Housing dataset
boston = load_boston()
X = boston.data[:, 5:6]  # Use one feature for simplicity (e.g., average number of rooms per dwelling)
y = boston.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Implement simple linear regression
linear_reg_model = LinearRegression()
linear_reg_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = linear_reg_model.predict(X_test)

# Visualize the linear regression line on a scatter plot
plt.scatter(X_test, y_test, color='blue', label='Actual Data')
plt.plot(X_test, y_pred, color='red', linewidth=3, label='Linear Regression Line')
plt.title('Simple Linear Regression')
plt.xlabel('Number of Rooms (X)')
plt.ylabel('House Price (y)')
plt.legend()
plt.show()
