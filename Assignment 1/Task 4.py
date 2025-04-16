import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import torch
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import logging

# Set up a basic logger
logger = logging.getLogger("MLLogger")
logger.setLevel(logging.DEBUG)  # Set the global logging level

# Setting up reusable template variables
prediction_col = 'mood'
feature_cols = [
    'activity', 'appCat.builtin', 'appCat.communication', 'appCat.entertainment',
    'appCat.finance', 'appCat.game', 'appCat.office', 'appCat.other',
    'appCat.social', 'appCat.travel', 'appCat.unknown', 'appCat.utilities',
    'appCat.weather', 'call', 'circumplex.arousal', 'circumplex.valence',
    'screen', 'sms'
]

data_sample = {
    'id': [0]*6,
    'time_bin': [1,2,3,4,5,6],
    'activity': [0.082]*6,
    'appCat.builtin': [0.0]*6,
    'appCat.communication': [0.0]*6,
    'appCat.entertainment': [0.0]*6,
    'appCat.finance': [0.0]*6,
    'appCat.game': [0.0]*6,
    'appCat.office': [0.0]*6,
    'appCat.other': [0.0]*6,
    'appCat.social': [0.0]*6,
    'appCat.travel': [0.0]*6,
    'appCat.unknown': [0.0]*6,
    'appCat.utilities': [0.0]*6,
    'appCat.weather': [0.0]*6,
    'call': [1.0]*6,
    'circumplex.arousal': [-1.0]*6,
    'circumplex.valence': [0.5]*6,
    'mood': [6.0]*6,
    'screen': [0.0]*6,
    'sms': [0.0]*6,
}

# Task 4

# Load the data
input_data = r"../input/df_interp_6hour.csv"
df = pd.read_csv(input_data)

# Define features and target
X = df.drop(columns=[prediction_col])
y = df[prediction_col]

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

# Task 4.Temporal

# Configure GPU transcoding if it is available, otherwise fall back to using just the cpu
device = torch.device("cpu")
print(f"Using device: {device}")

# Load the data
input_data = r"../input/df_interp_6hour.csv"
df = pd.read_csv(input_data)

# Perform scaling and sort data
scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])
df = df.sort_values(by=['id', 'time_bin']) # Scale by first user then time

# LSTM expects sequences, declare the amount
seq_length = 6  # (sequence length)
X_sequences = []
y_sequences = []

# Create sequences
for i in range(seq_length, len(df)):
    X_sequences.append(df[feature_cols].iloc[i-seq_length:i].values)  # Collect features for the sequence
    y_sequences.append(df['mood'].iloc[i])

X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_sequences, test_size=0.2, random_state=42)

# Define LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))  # LSTM layer
model.add(Dropout(0.2))  # Dropout for regularization
model.add(Dense(units=1, activation='linear'))  # Output layer for regression (mood prediction)

# Compile model with mean absolute error
model.compile(optimizer=Adam(), loss='mean_absolute_error', metrics=['mae'])

# Train model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Evaluate model
loss, mae = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test MAE: {mae}')

# Make predictions
predictions = model.predict(X_test)

# Evaluate with MAE for predictions
mae_value = mean_absolute_error(y_test, predictions)
print(f'Mean Absolute Error: {mae_value}')




# TODO Tune variables!!!

# # Optional: scale features
# pipeline = Pipeline([
#     ('scaler', StandardScaler()),
#     ('regressor', MLPRegressor(max_iter=2000, random_state=42))
# ])
#
# # Define hyperparameter grid for neural network
# param_grid = {
#     'regressor__hidden_layer_sizes': [(32,), (64,), (64, 32), (128, 64)],  # Number of neurons per layer
#     'regressor__activation': ['relu', 'tanh'],  # Activation functions
#     'regressor__solver': ['adam', 'lbfgs'],  # Optimizer types
#     'regressor__alpha': [0.0001, 0.001, 0.01],  # Regularization
# }
#
# # Perform grid search with 5-fold cross-validation (based on R² score)
# grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)
# grid_search.fit(X_train, y_train)
#
# # Use the best model found
# best_nn_model = grid_search.best_estimator_
# y_pred_nn = best_nn_model.predict(X_test)
#
# # Evaluate
# mse_nn = mean_squared_error(y_test, y_pred_nn)
# rmse_nn = np.sqrt(mse_nn)
# r2_nn = r2_score(y_test, y_pred_nn)
#
# # Tune the neural network regressor
# nn_model = MLPRegressor(hidden_layer_sizes=(4, 2),
#                         activation='relu',
#                         solver='adam',
#                         max_iter=500,
#                         random_state=42)
# nn_model.fit(X_train, y_train)
# y_pred_nn = nn_model.predict(X_test)
#
# mse_nn = mean_squared_error(y_test, y_pred_nn)
# rmse_nn = np.sqrt(mse_nn)
# r2_nn = r2_score(y_test, y_pred_nn)
#
# print(f"\nNeural Network Results:")
# print(f"Mean Squared Error (MSE): {mse_nn:.3f}")
# print(f"Root Mean Squared Error (RMSE): {rmse_nn:.3f}")
# print(f"R² Score: {r2_nn:.3f}")