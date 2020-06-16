# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# training the Decision tree regression model
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

# predicting the salary  for 6.5 level
print(regressor.predict([[6.5]]))

# visualizing the decision tree in high resolution
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y)
plt.plot(X_grid, regressor.predict(X_grid))
plt.title("Truth or Bluff (Decision Tree Regression)")
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
