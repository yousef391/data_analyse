import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
df=pd.read_csv(r'student_scores.csv')


X=df[['Hours']]
y=df['Scores']

print(X)
print('//////////')
print(df['Hours'])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 
model = LinearRegression()

model.fit(X_train, y_train)


intercept = model.intercept_
slope = model.coef_[0]
print(f"Model Equation: y = {slope:.2f} * X + {intercept:.2f}")

# 4. Create a plot showing the data and the generated model
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, model.predict(X), color='red', label='Regression line')
plt.xlabel('Hours Studied')
plt.ylabel('Scores')
plt.title('Hours Studied vs. Percentage Score')
plt.legend()
plt.show()

# 5
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate metrics
mae_train = mean_absolute_error(y_train, y_train_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(y_train, y_train_pred)

mae_test = mean_absolute_error(y_test, y_test_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test, y_test_pred)

print("Performance on Training Set:")
print(f"  Mean Absolute Error (MAE): {mae_train:.2f}")
print(f"  Mean Squared Error (MSE): {mse_train:.2f}")
print(f"  Root Mean Squared Error (RMSE): {rmse_train:.2f}")
print(f"  R-squared: {r2_train:.2f}")

print("Performance on Test Set:")
print(f"  Mean Absolute Error (MAE): {mae_test:.2f}")
print(f"  Mean Squared Error (MSE): {mse_test:.2f}")
print(f"  Root Mean Squared Error (RMSE): {rmse_test:.2f}")
print(f"  R-squared: {r2_test:.2f}")


# 7
if abs(r2_train - r2_test) < 0.1:
    print("The model does not present an overfitting problem.")
else:
    print("The model might present an overfitting problem.")

# 8. 
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, color='purple')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', color='red')
plt.xlabel('Actual Scores')
plt.ylabel('Predicted Scores')
plt.title('Actual vs. Predicted Scores')
plt.show()
