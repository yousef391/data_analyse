import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df=pd.read_csv(r'HousingData.csv')
print(df)

corr_matrix = df.corr()
print(corr_matrix)

high_correlations = corr_matrix['MEDV'][abs(corr_matrix['MEDV'])>0.6].index
print(high_correlations)

for column in high_correlations:
    df[column]=df[column].fillna(df[column].mean())






feature1, feature2 = high_correlations[0], high_correlations[1]  
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df[feature1], df[feature2], df['MEDV'], c='b', marker='o')
ax.set_xlabel(feature1)
ax.set_ylabel(feature2)
ax.set_zlabel('MEDV')
plt.show()


X = df[high_correlations.drop('MEDV')]
y = df['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_test)
print(y_test)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print('///////////////////')
print(X_train_scaled)
# Check the output


from sklearn.linear_model import SGDRegressor


learning_rates = [0.01, 0.001, 0.0001,0.005]
losses = {} 
metrics = {}

for lr in learning_rates:
    sgdr = SGDRegressor(learning_rate='invscaling', eta0=lr, max_iter=1, tol=None, warm_start=True)
    current_losses = []
    
    for i in range(1000):  # Manually iterate up to 1000 times
        sgdr.partial_fit(X_train_scaled, y_train)  # Perform one iteration
        y_pred = sgdr.predict(X_train_scaled)
        mse = mean_squared_error(y_train, y_pred)
        current_losses.append(mse)
        
        if len(current_losses) > 1 and abs(current_losses[-1] - current_losses[-2]) < 1e-6:
            break  # Early stopping if loss change is very small

    losses[lr] = current_losses
    y_test_pred = sgdr.predict(X_test_scaled)
    
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

# Plot convergence graphs
plt.figure(figsize=(10, 6))
for lr, loss in losses.items():
    plt.plot(loss, label=f'Learning Rate: {lr}')
plt.xlabel('Iterations')
plt.ylabel('Loss (MSE)')
plt.title('Convergence of SGDRegressor with Different Learning Rates')
plt.legend()
plt.show()


best_lr = min(metrics, key=lambda lr: metrics[lr]['MSE'])
print(f"Best model: Learning Rate = {best_lr}")


sgdr_best = SGDRegressor(learning_rate='invscaling', eta0=best_lr, max_iter=1000)
sgdr_best.fit(X_train_scaled, y_train)


x1_range = np.linspace(X_train_scaled[:, 0].min(), X_train_scaled[:, 0].max(), 100) #Feauture 1
x2_range = np.linspace(X_train_scaled[:, 1].min(), X_train_scaled[:, 1].max(), 100) #Feauture 2
#hadi derna bach n9dro n5rj l fueature w nprintohom

x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
grid_points = np.c_[x1_grid.ravel(), x2_grid.ravel()]


y_grid_pred = sgdr_best.predict(grid_points)
y_grid_pred = y_grid_pred.reshape(x1_grid.shape)

# Plotting the 3D surface plot
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], y_train, color='blue', label='Training Data')

# Plot the generated plane (model predictions) over the grid
ax.plot_surface(x1_grid, x2_grid, y_grid_pred, alpha=0.5, cmap='hot_r')

# Labels and title
ax.set_xlabel('RM')
ax.set_ylabel('LSTAT')
ax.set_zlabel('MEDV')
ax.set_title('Generated Plane and Training Data')
plt.legend()
plt.show()


sample = np.array([[5.713, 22.6]])

sample_sacled= scaler.transform(sample)
predicted_class = sgdr_best.predict(sample_sacled)
print(f"Predicted class for the sample: {predicted_class}")






