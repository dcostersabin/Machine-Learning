# import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# liner regression model
linear_reg = LinearRegression()
linear_reg.fit(X, y)

# polynomial regression model
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

# Visualizing the linear regression model's  and polynomial regression model's results
plt.scatter(X, y, edgecolors='red')
# linear model visualization
plt.plot(X, linear_reg.predict(X))
# polynomial regression visualization
plt.plot(X, lin_reg2.predict(X_poly))
plt.title('Truth Or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# predicting the salary for 6.5
# using linear model
print("Expected Salary %f using linear model" % linear_reg.predict([[6.5]]))
# using polynomial model
print("Expected Salary %f using polynomial model" % lin_reg2.predict(poly_reg.fit_transform([[6.5]])))
