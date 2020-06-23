import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# importing the dataset
datset = pd.read_csv('Position_Salaries.csv')
X = datset.iloc[:, 1:-1].values
y = datset.iloc[:, -1].values

# training the model
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X, y)
print(regressor.predict([[6.5]]))

# visualizing the decision tree in high resolution
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y)
plt.plot(X_grid, regressor.predict(X_grid))
plt.title("Truth or Bluff (Random Forest Regression)")
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
