import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold, KFold
import logging
import keras_tuner as kt
import optuna

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

## == Task 2A == ##
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

# 1. First pass: GridSearch to get good starting point
baseline_model = RandomForestClassifier(random_state=42)

search_space = {
    'criterion': ['gini', 'entropy'],
    'n_estimators': range(50, 201, 25),
    'max_depth': range(5, 26, 5),
    'min_samples_split': range(2,5),
    'min_samples_leaf': range(1,5),
    'max_features': ["sqrt", "log2"]
}

# Perform grid search with 3-fold cross-validation (based on accuracy score)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

print("Running initial GridSearch...")
tuned_model = GridSearchCV(baseline_model, search_space, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
fitted_model = tuned_model.fit(X_train, y_train)
best_grid_params = fitted_model.best_params_
print("Initial GridSearch best hyperparameters:", str(best_grid_params))

# 2. Second pass: Optuna refinement around GridSearch result
def objective(trial):
    params = {
        "criterion": trial.suggest_categorical("criterion", [best_grid_params['criterion']]),
        "n_estimators": trial.suggest_int("n_estimators", max(45, best_grid_params['n_estimators'] - 10), best_grid_params['n_estimators'] + 10),
        "max_depth": trial.suggest_int("max_depth", max(2, best_grid_params['max_depth'] - 3), best_grid_params['max_depth'] + 3),
        "min_samples_split": trial.suggest_int("min_samples_split", max(2, best_grid_params['min_samples_split'] - 1), best_grid_params['min_samples_split'] + 1),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", max(1, best_grid_params['min_samples_leaf'] - 1), best_grid_params['min_samples_leaf'] + 1),
        "max_features": trial.suggest_categorical("max_features", [best_grid_params['max_features']]),
        "random_state": 42,
        "n_jobs": -1,
    }

    clf = RandomForestClassifier(**params)
    score = cross_val_score(clf, X_train, y_train, cv=cv, scoring="accuracy")
    return score.mean()

print("Running Optuna tuning...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=40)

optuna_best_params = study.best_params
print("Optuna best:", optuna_best_params)
print("Optuna best score:", study.best_value)

# Train final model with Optuna best params
final_model = RandomForestClassifier(**optuna_best_params)
final_model.fit(X_train, y_train)

# Evaluation
y_pred = final_model.predict(X_test)
print("Final best parameters:", optuna_best_params)
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# Feature importances
feature_importances = pd.Series(final_model.feature_importances_, index=X.columns)
feature_importances.nlargest(10).plot(kind='barh')
plt.title("Top 10 Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()



## == Task 2A.Temporal == ##

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

# Hyperparameter search help function
def build_model(hp):
    model = Sequential()
    model.add(LSTM(
        units=hp.Int('units', min_value=32, max_value=96, step=32),
        return_sequences=False,
        input_shape=(X_train.shape[1], X_train.shape[2])
    ))
    model.add(Dropout(hp.Choice('dropout_rate', [0.1, 0.2, 0.3, 0.5])))
    model.add(Dense(4, activation='softmax'))
    model.compile(
        optimizer=Adam(learning_rate=hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Optimum found by HP search: 0.985 accuracy, mae 0.76, loss 1.26. {'units': 96, 'dropout_rate': 0.1, 'learning_rate': 0.01}
tuner = kt.RandomSearch(
    build_model,
    objective='accuracy',
    max_trials=500,
    executions_per_trial=3,
    directory='',
    project_name='lstm_classification'
)

# Train model
tuner.search(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

best_model = tuner.get_best_models(num_models=1)[0]
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
print(best_hp.values)

# Evaluate model
loss, mae = best_model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test MAE: {mae}')

# Fine-tuning the hyperparameters with Optuna around Keras Tuner result
def objective(trial):
    units = trial.suggest_int('units', best_hp['units'] - 16,
       best_hp['units'] + 16)
    dropout_rate = trial.suggest_categorical('dropout_rate', [best_hp['dropout_rate'] - 0.05, best_hp['dropout_rate'],
       best_hp['dropout_rate'] + 0.05])
    learning_rate = trial.suggest_loguniform('learning_rate', best_hp['learning_rate'] * 0.1,
       best_hp['learning_rate'] * 10)

    model = Sequential()
    model.add(LSTM(units=units, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(dropout_rate))
    model.add(Dense(4, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # K-fold cross-validation
    kfold = KFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = []

    for train_idx, val_idx in kfold.split(X_train):
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        # Train the model on the fold
        model.fit(X_train_fold, y_train_fold, epochs=10, batch_size=32, verbose=0)

        # Evaluate on validation fold
        val_loss, val_accuracy = model.evaluate(X_val_fold, y_val_fold, verbose=0)
        cv_scores.append(val_accuracy)

    # Return the average accuracy score for this trial
    return np.mean(cv_scores)


# Create an Optuna study and optimize it
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
optuna_best_params = study.best_params
print("Best Optuna Hyperparameters:", optuna_best_params)

# Train the final model with the best Optuna hyperparameters
final_model = Sequential()
final_model.add(LSTM(units=optuna_best_params['units'], return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
final_model.add(Dropout(optuna_best_params['dropout_rate']))
final_model.add(Dense(4, activation='softmax'))

final_model.compile(optimizer=Adam(learning_rate=optuna_best_params['learning_rate']),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

final_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

#Evaluate the final model
loss, accuracy = final_model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# Make predictions
predictions = final_model.predict(X_test)
y_pred = np.argmax(predictions, axis=1)

print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
print("Accuracy Score:", accuracy_score(y_test, y_pred))
