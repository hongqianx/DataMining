import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error
import logging
import keras_tuner as kt

# Set up a basic logger
logger = logging.getLogger("MLLogger")
logger.setLevel(logging.DEBUG)  # Set the global logging level

# Configure GPU transcoding if it is available, otherwise fall back to using just the cpu (tested on cpu, gpu untested)
# Comment back in to use gpu
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# fix for tf always choosing the GPU instead of the CPU. Comment out to use gpu (untested)
tf.config.set_visible_devices([], 'GPU')

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
input_data = r"../input/df_rolling.csv"
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
model = RandomForestRegressor(random_state=42)

search_space = {
    'n_estimators': [50,100,150,200],
    'max_depth': [5, 10, 15, 20, 25],
    'min_samples_split': [2,3,4],
    'min_samples_leaf': [1,2,3],
    'max_features': ["sqrt", "log2"]
}

# Perform grid search with 3-fold cross-validation (based on R² score)
tuned_model = GridSearchCV(model, search_space, cv=3, scoring='r2', n_jobs=-1, verbose=1)

fitted_model = tuned_model.fit(X_train, y_train)
print("Best hyperparameters:", str(fitted_model.best_params_))

# Evaluate the model
y_pred = fitted_model.predict(X_test)
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))

# Feature importance plot
feature_importances = pd.Series(fitted_model.best_estimator_.feature_importances_, index=X.columns)
feature_importances.nlargest(10).plot(kind='barh')
plt.title("Top 10 Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# Task 4.Temporal

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

def build_model(hp):
    model = Sequential()
    model.add(LSTM(
        units=hp.Int('units', min_value=32, max_value=96, step=32),
        return_sequences=False,
        input_shape=(X_train.shape[1], X_train.shape[2])
    ))
    model.add(Dropout(hp.Choice('dropout_rate', [0.1, 0.2, 0.3, 0.5])))
    model.add(Dense(1, activation='linear'))
    model.compile(
        optimizer=Adam(learning_rate=hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='mean_absolute_error',
        metrics=['mae']
    )
    return model

# optimum: {'units': 96, 'dropout_rate': 0.1, 'learning_rate': 0.01}. best mae 0.5109, test loss= test mae = 0.586
tuner = kt.RandomSearch(
    build_model,
    objective='mae',
    max_trials=500,
    executions_per_trial=3,
    directory='',
    project_name='lstm_regression'
)

# Train model
tuner.search(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

best_model = tuner.get_best_models(num_models=1)[0]
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
print(best_hp.values)

# Evaluate model
loss, mae = best_model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test MAE: {mae}')

# Make predictions
predictions = best_model.predict(X_test)

# Evaluate with MAE for predictions
mae_value = mean_absolute_error(y_test, predictions)
print(f'Mean Absolute Error: {mae_value}')
