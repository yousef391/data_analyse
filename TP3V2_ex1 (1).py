import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
from io import StringIO

df=pd.read_csv(r'HousingData.csv')

# print(df)

corr_matrix = df.corr()
high_corr_features = corr_matrix.index[abs(corr_matrix["MEDV"]) > 0.6]
high_corr_matrix = corr_matrix.loc[high_corr_features, high_corr_features]


print("\nHighly Correlated Features with MEDV:")
print(high_corr_matrix)

feature_x = high_corr_features[0] 
feature_y = high_corr_features[1] 
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df[feature_x], df[feature_y], df['MEDV'], color='b', marker='o')
ax.set_xlabel(feature_x)
ax.set_ylabel(feature_y)
ax.set_zlabel('MEDV')
plt.title(f'3D Scatter Plot of {feature_x}, {feature_y}, and MEDV')
plt.show()



X = df[high_corr_features.drop('MEDV')]  
y = df['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

scaler = MinMaxScaler()
X_train_normalized = scaler.fit_transform(X_train_imputed) #calculate min and max and transform the data
X_test_normalized = scaler.transform(X_test_imputed)#tranform only


print("x_train shape:", X_train.shape)
print("x_test shape:", X_test.shape)
print("\nTraining features shape:", X_train_normalized.shape)
print("Testing features shape:", X_test_normalized.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

learning_rates = [0.01, 0.001, 0.0001]

metrics = {}
loss_histories = {}

# Train the model for each learning rate
for lr in learning_rates:
    old_stdout = sys.stdout # saves the original terminal
    sys.stdout = mystdout = StringIO() # any print statement within this context will be directed to mystdout instead of being printed to the console.
    
    sgdr = SGDRegressor(learning_rate='invscaling', eta0=lr, max_iter=1000, verbose=1)
    sgdr.fit(X_train_normalized, y_train)
    
    # Restore stdout and retrieve loss history
    sys.stdout = old_stdout
    loss_history = mystdout.getvalue()
    
    # Extract loss values from verbose output
    loss_list = []
    for line in loss_history.split('\n'):
        if "loss: " in line:
            loss_list.append(float(line.split("loss: ")[-1]))
    
    # Store the loss history for each learning rate
    loss_histories[lr] = loss_list

    y_test_pred = sgdr.predict(X_test_normalized)
    
    mae = mean_absolute_error(y_test, y_test_pred)
    mse = mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_test_pred)
    
    metrics[lr] = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R^2': r2
    }
    print(f"Learning rate {lr}:\n MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R^2: {r2}\n")

plt.figure(figsize=(10, 6))
for lr, loss_list in loss_histories.items():
    plt.plot(np.arange(len(loss_list)), loss_list, label=f"Learning Rate {lr}")
plt.title("Loss over Iterations for Different Learning Rates")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()



best_lr = max(metrics, key=lambda lr: metrics[lr]['R^2'])
print(f"Best model: Learning Rate = {best_lr}")


sgdr_best = SGDRegressor(learning_rate='invscaling', eta0=best_lr, max_iter=1000)
sgdr_best.fit(X_train_normalized, y_train)


x1_range = np.linspace(X_train_normalized[:, 0].min(), X_train_normalized[:, 0].max(), 100)
x2_range = np.linspace(X_train_normalized[:, 1].min(), X_train_normalized[:, 1].max(), 100)
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
grid_points = np.c_[x1_grid.ravel(), x2_grid.ravel()]



y_grid_pred = sgdr_best.predict(grid_points).reshape(x1_grid.shape)
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train_normalized[:, 0], X_train_normalized[:, 1], y_train, color='g', label='Training Data')
ax.plot_surface(x1_grid, x2_grid, y_grid_pred, alpha=0.5, cmap='Blues')


# Set labels and title
ax.set_xlabel('RM')
ax.set_ylabel('LSTAT')
ax.set_zlabel('MEDV')
ax.set_title('Generated Plane and Training Data')
plt.legend()
plt.show()

# tol : The tolerance for early stopping is set by the tol option. 
# If the improvement in the loss function between iterations is less than this value, training will end. 
# By terminating training early when more optimization does not yield a discernible improvement in the model, 
# one can minimize training time and avoid overfitting.


sample_data = {
    'RM': [5.713],
    'LSTAT': [22.6]
}

sample_df = pd.DataFrame(sample_data)
predicted_class = sgdr_best.predict(sample_df)
print(f"Predicted class for the sample: {predicted_class}")