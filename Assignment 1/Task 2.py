import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import torch
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

# Define function to split our mood in segments
# For now we split it in 0 = low, 1 = neutral, 2 = good, 3 = great
def split_mood_segments(mood):
    match mood:
        case mood if mood <= 4:
            return 0
        case mood if mood <= 6:
            return 1
        case mood if mood <= 8:
            return 2
        case mood if mood > 8:
            return 3
        case _:
            logger.error(f"Invalid mood value: {mood}")
            return -1

# Setting up reusable template variables
prediction_col = 'mood'
feature_cols = [
    'activity', 'appCat.builtin', 'appCat.communication', 'appCat.entertainment',
    'appCat.finance', 'appCat.game', 'appCat.office', 'appCat.other',
    'appCat.social', 'appCat.travel', 'appCat.unknown', 'appCat.utilities',
    'appCat.weather', 'call', 'circumplex.arousal', 'circumplex.valence',
    'screen', 'sms'
]

# Task 2A

# Load the data
input_data = r"../input/df_rolling.csv"
df = pd.read_csv(input_data)

# Define features and target
X = df.drop(columns=[prediction_col])
y = df[prediction_col].apply(lambda x: split_mood_segments(x))

# Handle missing values
X = X.fillna(0)
y = y.fillna(y.mean())

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestClassifier(random_state=42, max_depth=15, max_features='sqrt', min_samples_leaf=3, min_samples_split=2, n_estimators=150)

model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Feature importance plot
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances.nlargest(10).plot(kind='barh')
plt.title("Top 10 Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# Task 2A.Temporal

# Configure GPU transcoding if it is available, otherwise fall back to using just the cpu (tested on cpu, gpu untested)
# Comment back in to use gpu
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# fix for tf always choosing the GPU instead of the CPU. Comment out to use gpu (untested)
tf.config.set_visible_devices([], 'GPU')

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
y_sequences = np.array([split_mood_segments(mood) for mood in y_sequences])

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_sequences, test_size=0.2, random_state=42)

# Define LSTM model prefilled with best values found in hyperparameter search.
model = Sequential()
model.add(LSTM(units=96, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))  # LSTM layer
model.add(Dropout(0.1))  # Dropout for regularization
model.add(Dense(units=4, activation='softmax')) # Output layer for classification (4 mood labels)
# Compile model with cross entropy since we have more than 2 labels
model.compile(optimizer=Adam(learning_rate=0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Evaluate model
loss, mae = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test MAE: {mae}')

# Make predictions
predictions = model.predict(X_test)
y_pred = np.argmax(predictions, axis=1)

print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
print("Accuracy Score:", accuracy_score(y_test, y_pred))
