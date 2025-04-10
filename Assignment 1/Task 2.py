import pandas as pd
from sklearn import pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Task 2A

# Load the data
input_data = r"../input/df_interp_6hour.csv"
df = pd.read_csv(input_data)

# Define features and target
X = df.drop(columns=['mood'])
y = df['mood']

# Handle missing values
X = X.fillna(0)
y = y.fillna(y.mean())

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Feature importance plot
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances.nlargest(10).plot(kind='barh')
plt.title("Top 10 Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# Task 2A.Temporal

# Load the data
input_data = r"../input/df_interp_6hour.csv"
df = pd.read_csv(input_data)

# TODO Tune variables!!!

# Optional: scale features
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', MLPRegressor(max_iter=2000, random_state=42))
])

# Define hyperparameter grid for neural network
param_grid = {
    'regressor__hidden_layer_sizes': [(32,), (64,), (64, 32), (128, 64)],  # Number of neurons per layer
    'regressor__activation': ['relu', 'tanh'],  # Activation functions
    'regressor__solver': ['adam', 'lbfgs'],  # Optimizer types
    'regressor__alpha': [0.0001, 0.001, 0.01],  # Regularization
}

# Perform grid search with 5-fold cross-validation (based on R² score)
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Use the best model found
best_nn_model = grid_search.best_estimator_
y_pred_nn = best_nn_model.predict(X_test)

# Evaluate
mse_nn = mean_squared_error(y_test, y_pred_nn)
rmse_nn = np.sqrt(mse_nn)
r2_nn = r2_score(y_test, y_pred_nn)

# Tune the neural network regressor
nn_model = MLPRegressor(hidden_layer_sizes=(4, 2),
                        activation='relu',
                        solver='adam',
                        max_iter=500,
                        random_state=42)
nn_model.fit(X_train, y_train)
y_pred_nn = nn_model.predict(X_test)

mse_nn = mean_squared_error(y_test, y_pred_nn)
rmse_nn = np.sqrt(mse_nn)
r2_nn = r2_score(y_test, y_pred_nn)

print(f"\nNeural Network Results:")
print(f"Mean Squared Error (MSE): {mse_nn:.3f}")
print(f"Root Mean Squared Error (RMSE): {rmse_nn:.3f}")
print(f"R² Score: {r2_nn:.3f}")