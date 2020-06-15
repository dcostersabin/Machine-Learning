# importing  libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# feature scaling
y = y.reshape(len(y), 1)
sc_x = StandardScaler()
X = sc_x.fit_transform(X)
sc_y = StandardScaler()
y = sc_y.fit_transform(y)

# training the model
regressor = SVR(kernel='rbf')
regressor.fit(X, y)

# predicting the result
print(sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]]))))

# visualizing the results
plt.scatter(sc_x.inverse_transform(X), sc_y.inverse_transform(y))
plt.plot(sc_x.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X)))
plt.title('Truth or Bluff (Support Verctor Regression)')
plt.xlabel('Position or Level')
plt.ylabel('Salary')
plt.show()

# # high resolution curve
X_grid = np.arange(min(sc_x.inverse_transform(X)), max(sc_x.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_x.inverse_transform(X), sc_y.inverse_transform(y), edgecolors='red')
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_x.transform(X_grid))))
plt.title('Truth or Bluff (Support Verctor Regression) High Resolution')
plt.xlabel('Position or Level')
plt.ylabel('Salary')
plt.show()
