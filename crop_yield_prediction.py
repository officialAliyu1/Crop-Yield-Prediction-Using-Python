# Python Project: Crop Yield Prediction Using Regression

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Sample crop yield dataset
data = pd.DataFrame({
    'Rainfall': [100, 120, 130, 90, 110, 105, 140, 95, 125, 135],
    'Fertilizer': [50, 55, 60, 40, 52, 47, 65, 42, 58, 62],
    'Yield': [2.8, 3.1, 3.5, 2.5, 3.0, 2.7, 3.8, 2.6, 3.3, 3.6]
})

# Check for missing values
if data.isnull().values.any():
    print("Missing values found!")
    print(data.isnull().sum())

# Define features (X) and target variable (Y)
X = data[['Rainfall', 'Fertilizer']]
y = data['Yield']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Create a DataFrame for test data predictions here
test_data = X_test.copy()
test_data['Actual_Yield'] = y_test
test_data['Predicted_Yield'] = y_pred

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared Score: {r2}')

# Visualization
plt.scatter(test_data['Actual_Yield'], test_data['Predicted_Yield'], color='blue')
plt.plot([min(test_data['Actual_Yield']), max(test_data['Actual_Yield'])],
         [min(test_data['Actual_Yield']), max(test_data['Actual_Yield'])], linestyle='dashed', color='red')
plt.xlabel('Actual Yield')
plt.ylabel('Predicted Yield')
plt.title('Actual vs Predicted Crop Yield')
plt.show()

# Save results
test_data.to_csv('crop_yield_predictions_python.csv', index=False)
print("Crop yield predictions saved for reporting.")