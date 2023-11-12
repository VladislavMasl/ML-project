import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
################################################GRADIENT##BOOSTING###################################
file_path = '/ivium_new_data/tetracycline.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataframe to inspect the data
print(data.head())

# Assuming 'concentration' is your feature and 'E/A' is your target variable
X = data[['E/V']]
y = data['concentration']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
rf_model = GradientBoostingRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f'Root Mean Squared Error (RMSE): {rmse}')

plt.scatter(X_test, y_test, color='black', label='Actual')
plt.scatter(X_test, y_pred, color='red', label='Predicted')
plt.xlabel('I/A')
plt.ylabel('Concentration')
plt.legend()
plt.show()
##################################################RANDOM##FOREST##REGRESSION#################################
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Assuming your CSV file is in the same directory as your script or notebook
file_path = '/ivium_new_data/tetracycline.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataframe to inspect the data
print(data.head())

# Assuming 'concentration' is your feature and 'E/A' is your target variable
X = data[['E/V']]
y = data['concentration']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f'Root Mean Squared Error (RMSE): {rmse}')

import matplotlib.pyplot as plt

plt.scatter(X_test, y_test, color='black', label='Actual')
plt.scatter(X_test, y_pred, color='red', label='Predicted')
plt.xlabel('I/A')
plt.ylabel('Concentration')
plt.legend()
plt.show()

##################################################DECISION##TREE###33#################################
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

file_path = '/ivium_new_data/tetracycline.csv'
data = pd.read_csv(file_path)
print(data.head())

# Assuming 'concentration' is your feature and 'E/A' is your target variable
X = data[['E/V']]
y = data['concentration']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree Regressor
dt_model = DecisionTreeRegressor(random_state=42)

# Train the model
dt_model.fit(X_train, y_train)

# Make predictions on the test set
y_dt_pred = dt_model.predict(X_test)

# Evaluate the model using Root Mean Squared Error (RMSE)
rmse_dt = mean_squared_error(y_test, y_dt_pred, squared=False)
print(f'Decision Tree Root Mean Squared Error (RMSE): {rmse_dt}')

import matplotlib.pyplot as plt

plt.scatter(X_test, y_test, color='black', label='Actual')
plt.scatter(X_test, y_dt_pred, color='blue', label='Decision Tree Predicted')
plt.xlabel('I/A')
plt.ylabel('Concentration')
plt.legend()
plt.show()
