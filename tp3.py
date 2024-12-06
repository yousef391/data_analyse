import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
df=pd.read_csv(r'student_scores.csv')


X=df[['Hours']]
y=df['Scores']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Train the data using the LinearRegression module from sklearn
model = LinearRegression()
try:
    model.fit(X_train, y_train)
except Exception as e:
    print("Error during fitting:", e)

print(X.shape , y.shape)

model=LinearRegression()

# 3. Display the model's equation









