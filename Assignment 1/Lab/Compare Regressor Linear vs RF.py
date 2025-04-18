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
import logging
import keras_tuner as kt
import os
import time

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RegressorComparisonScript")

try:
    physical_devices_gpu = tf.config.list_physical_devices('GPU')
    if physical_devices_gpu:
        tf.config.set_visible_devices([], 'GPU')
        logger.info("TensorFlow GPU usage disabled.")
    else:
        logger.info("No GPU found or TensorFlow GPU support not available. Using CPU.")
except Exception as e:
    logger.warning(f"Could not configure TensorFlow devices: {e}")

# --- File Paths (IMPORTANT: Update these paths if necessary) ---
rolling_data_path = "../../input/df_rolling.csv"
interp_data_path = "../../input/df_interp_6hour.csv"

if not os.path.exists(rolling_data_path):
    logger.error(f"Rolling data file not found at: {rolling_data_path}")
if not os.path.exists(interp_data_path):
    logger.error(f"Interpolated data file not found at: {interp_data_path}")

# --- Reusable Variables ---
prediction_col = 'mood'
feature_cols = [
    'activity', 'appCat.builtin', 'appCat.communication', 'appCat.entertainment',
    'appCat.finance', 'appCat.game', 'appCat.office', 'appCat.other',
    'appCat.social', 'appCat.travel', 'appCat.unknown', 'appCat.utilities',
    'appCat.weather', 'call', 'circumplex.arousal', 'circumplex.valence',
    'screen', 'sms'
]
random_seed = 42
test_size = 0.2
results_reg = {} # Dictionary specific for regressor results

# --- 1. Random Forest Regressor ---
logger.info("--- Starting: Random Forest Regressor ---")
start_time_rf = time.time()
try:
    df_reg_rf = pd.read_csv(rolling_data_path)
    X_reg_rf = df_reg_rf.drop(columns=[prediction_col], errors='ignore').fillna(0)
    if prediction_col not in df_reg_rf.columns: raise ValueError(f"Target column '{prediction_col}' not found in {rolling_data_path}")
    y_reg_rf = df_reg_rf[prediction_col].fillna(df_reg_rf[prediction_col].mean())

    # Align features
    missing_cols_train = set(feature_cols) - set(X_reg_rf.columns)
    if missing_cols_train:
        logger.warning(f"RF Regressor: Missing columns {missing_cols_train}. Filling with 0.")
        for col in missing_cols_train: X_reg_rf[col] = 0
    extra_cols_train = set(X_reg_rf.columns) - set(feature_cols)
    if extra_cols_train:
         logger.warning(f"RF Regressor: Extra columns {extra_cols_train}. Dropping them.")
         X_reg_rf = X_reg_rf.drop(columns=list(extra_cols_train))
    X_reg_rf = X_reg_rf[[col for col in feature_cols if col in X_reg_rf.columns]] # Use only available & specified cols

    X_train_reg_rf, X_test_reg_rf, y_train_reg_rf, y_test_reg_rf = train_test_split(X_reg_rf, y_reg_rf, test_size=test_size, random_state=random_seed)

    # Hyperparameter Tuning
    model_reg_rf = RandomForestRegressor(random_state=random_seed)
    # Reduced search space for quicker execution in a combined script
    search_space_reg_rf = {'n_estimators': [50, 100], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 3], 'max_features': ["sqrt", "log2"]}
    tuned_model_reg_rf = GridSearchCV(model_reg_rf, search_space_reg_rf, cv=3, scoring='r2', n_jobs=-1, verbose=0)
    logger.info("Fitting RF Regressor GridSearchCV...")
    fitted_model_reg_rf = tuned_model_reg_rf.fit(X_train_reg_rf, y_train_reg_rf)
    logger.info(f"RF Regressor Best Hyperparameters: {fitted_model_reg_rf.best_params_}")

    # Evaluate
    final_model_reg_rf = fitted_model_reg_rf.best_estimator_
    y_pred_reg_rf = final_model_reg_rf.predict(X_test_reg_rf)
    rf_mae = mean_absolute_error(y_test_reg_rf, y_pred_reg_rf)
    rf_mse = mean_squared_error(y_test_reg_rf, y_pred_reg_rf)
    rf_r2 = r2_score(y_test_reg_rf, y_pred_reg_rf)

    # Store results
    results_reg['RF_Regressor'] = {
        'MAE': rf_mae, 'MSE': rf_mse, 'R2': rf_r2, 'Error': None,
        'Best Params': fitted_model_reg_rf.best_params_,
        'Feature Importances': pd.Series(final_model_reg_rf.feature_importances_, index=X_reg_rf.columns)
    }
    logger.info(f"RF Regressor Metrics - MAE: {rf_mae:.4f}, MSE: {rf_mse:.4f}, R2: {rf_r2:.4f}")

    # Plot Feature Importance
    plt.figure(figsize=(8, 6))
    results_reg['RF_Regressor']['Feature Importances'].nlargest(10).plot(kind='barh')
    plt.title("RF Regressor: Top 10 Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("rf_regressor_feature_importance.png")
    logger.info("Saved RF Regressor feature importance plot.")
    plt.close()

except FileNotFoundError:
    logger.error(f"Skipping RF Regressor: Data file not found at {rolling_data_path}")
    results_reg['RF_Regressor'] = {'Error': 'Data file not found'}
except Exception as e:
    logger.error(f"Error during RF Regressor execution: {e}", exc_info=True)
    results_reg['RF_Regressor'] = {'Error': str(e)}
end_time_rf = time.time()
logger.info(f"RF Regressor took {end_time_rf - start_time_rf:.2f} seconds.")


# --- 2. LSTM Regressor ---
logger.info("--- Starting: LSTM Regressor ---")
start_time_lstm = time.time()
try:
    df_reg_lstm = pd.read_csv(interp_data_path)
    scaler_reg_lstm = StandardScaler()
    cols_to_scale_reg = [col for col in feature_cols if col in df_reg_lstm.columns]
    available_feature_cols_lstm_reg = cols_to_scale_reg
    if len(available_feature_cols_lstm_reg) < len(feature_cols): logger.warning(f"LSTM Regressor: Using subset of features due to missing columns in {interp_data_path}: {available_feature_cols_lstm_reg}")
    if not available_feature_cols_lstm_reg: raise ValueError("No feature columns found for LSTM Regressor.")
    df_reg_lstm[available_feature_cols_lstm_reg] = scaler_reg_lstm.fit_transform(df_reg_lstm[available_feature_cols_lstm_reg])
    df_reg_lstm = df_reg_lstm.sort_values(by=['id', 'time_bin'])

    # Create sequences
    seq_length_reg = 6
    X_seq_reg_lstm, y_seq_reg_lstm = [], []
    if prediction_col not in df_reg_lstm.columns: raise ValueError(f"Target column '{prediction_col}' not found in {interp_data_path}")
    df_reg_lstm[prediction_col] = df_reg_lstm[prediction_col].fillna(df_reg_lstm[prediction_col].mean())
    grouped = df_reg_lstm.groupby('id')
    for group_id, group in grouped:
        if len(group) >= seq_length_reg:
            for i in range(seq_length_reg, len(group)):
                X_seq_reg_lstm.append(group[available_feature_cols_lstm_reg].iloc[i - seq_length_reg:i].values)
                y_seq_reg_lstm.append(group[prediction_col].iloc[i])
        else: logger.warning(f"Skipping id {group_id} for LSTM Regressor sequencing: too few records ({len(group)} < {seq_length_reg})")
    if not X_seq_reg_lstm: raise ValueError("No sequences created for LSTM Regressor. Check data, grouping, and seq_length.")
    X_seq_reg_lstm, y_seq_reg_lstm = np.array(X_seq_reg_lstm), np.array(y_seq_reg_lstm)
    if X_seq_reg_lstm.shape[1] != seq_length_reg or X_seq_reg_lstm.shape[2] != len(available_feature_cols_lstm_reg): raise ValueError(f"Unexpected sequence shape for LSTM Regressor: {X_seq_reg_lstm.shape}. Expected ({len(X_seq_reg_lstm)}, {seq_length_reg}, {len(available_feature_cols_lstm_reg)})")
    X_train_reg_lstm, X_test_reg_lstm, y_train_reg_lstm, y_test_reg_lstm = train_test_split(X_seq_reg_lstm, y_seq_reg_lstm, test_size=test_size, random_state=random_seed)

    # Hyperparameter Tuning
    def build_model_reg(hp):
        model = Sequential([
            LSTM(units=hp.Int('units', 32, 96, step=32), return_sequences=False, input_shape=(X_train_reg_lstm.shape[1], X_train_reg_lstm.shape[2])),
            Dropout(hp.Choice('dropout_rate', [0.1, 0.3, 0.5])),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=hp.Choice('learning_rate', [1e-3, 1e-4])), loss='mean_absolute_error', metrics=['mae'])
        return model
    # Reduced trials/epochs for quicker execution
    tuner_reg_lstm = kt.RandomSearch(build_model_reg, objective='val_mae', max_trials=5, executions_per_trial=1, directory='kt_reg_dir', project_name='lstm_regression_compare', overwrite=True)
    logger.info("Searching LSTM Regressor hyperparameters...")
    tuner_reg_lstm.search(X_train_reg_lstm, y_train_reg_lstm, epochs=8, batch_size=64, validation_data=(X_test_reg_lstm, y_test_reg_lstm), verbose=0, callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=3)])
    best_hp_reg_lstm = tuner_reg_lstm.get_best_hyperparameters(num_trials=1)[0]
    logger.info(f"LSTM Regressor Best Hyperparameters: {best_hp_reg_lstm.values}")

    # Evaluate
    best_model_reg_lstm = tuner_reg_lstm.get_best_models(num_models=1)[0]
    lstm_loss, lstm_mae = best_model_reg_lstm.evaluate(X_test_reg_lstm, y_test_reg_lstm, verbose=0)
    y_pred_reg_lstm = best_model_reg_lstm.predict(X_test_reg_lstm).flatten()
    lstm_mse = mean_squared_error(y_test_reg_lstm, y_pred_reg_lstm)
    lstm_r2 = r2_score(y_test_reg_lstm, y_pred_reg_lstm)

    # Store results
    results_reg['LSTM_Regressor'] = {
        'MAE': lstm_mae, 'MSE': lstm_mse, 'R2': lstm_r2, 'Error': None,
         'Loss': lstm_loss, 'Best Params': best_hp_reg_lstm.values
    }
    logger.info(f"LSTM Regressor Metrics - MAE: {lstm_mae:.4f}, MSE: {lstm_mse:.4f}, R2: {lstm_r2:.4f}")

except FileNotFoundError:
    logger.error(f"Skipping LSTM Regressor: Data file not found at {interp_data_path}")
    results_reg['LSTM_Regressor'] = {'Error': 'Data file not found'}
except Exception as e:
    logger.error(f"Error during LSTM Regressor execution: {e}", exc_info=True)
    results_reg['LSTM_Regressor'] = {'Error': str(e)}
end_time_lstm = time.time()
logger.info(f"LSTM Regressor took {end_time_lstm - start_time_lstm:.2f} seconds.")


# --- 3. Comparison Logic ---
logger.info("--- Comparing LSTM Regressor vs. Random Forest Regressor ---")

print("\n" + "="*60)
print(" Comparison: LSTM Regressor vs. Random Forest Regressor")
print(" (Based on Regression Metrics Only)")
print("="*60 + "\n")

# Safely retrieve results from this run
lstm_res = results_reg.get('LSTM_Regressor', {})
rf_res = results_reg.get('RF_Regressor', {})

lstm_mae = lstm_res.get('MAE', float('inf')) # Use MAE from results
lstm_mse = lstm_res.get('MSE', float('inf'))
lstm_r2 = lstm_res.get('R2', float('-inf'))
lstm_error = lstm_res.get('Error')

rf_mae = rf_res.get('MAE', float('inf'))
rf_mse = rf_res.get('MSE', float('inf'))
rf_r2 = rf_res.get('R2', float('-inf'))
rf_error = rf_res.get('Error')

# --- Print Metrics ---
print(f"Metric                 | LSTM Regressor        | RF Regressor")
print(f"-----------------------|-----------------------|-----------------------")

if lstm_error and rf_error:
    print("MAE (lower is better)  | Error                 | Error")
    print("MSE (lower is better)  | Error                 | Error")
    print("R2 (higher is better)  | Error                 | Error")
elif lstm_error:
    print(f"MAE (lower is better)  | Error                 | {rf_mae:.4f}")
    print(f"MSE (lower is better)  | Error                 | {rf_mse:.4f}")
    print(f"R2 (higher is better)  | Error                 | {rf_r2:.4f}")
elif rf_error:
    print(f"MAE (lower is better)  | {lstm_mae:.4f}                | Error")
    print(f"MSE (lower is better)  | {lstm_mse:.4f}                | Error")
    print(f"R2 (higher is better)  | {lstm_r2:.4f}                | Error")
else:
    print(f"MAE (lower is better)  | {lstm_mae:.4f}                | {rf_mae:.4f}")
    print(f"MSE (lower is better)  | {lstm_mse:.4f}                | {rf_mse:.4f}")
    print(f"R2 (higher is better)  | {lstm_r2:.4f}                | {rf_r2:.4f}")

# --- Insight and Conclusion ---
print("\n" + "-"*60)
print("Insight (Based on Regression Performance):")

valid_lstm = not lstm_error and lstm_mae != float('inf')
valid_rf = not rf_error and rf_mae != float('inf')

if valid_lstm and valid_rf:
    # Compare based on MAE primarily
    if abs(lstm_mae - rf_mae) < 0.001:
        print("  - Performance is very similar based on MAE (~{:.4f}).".format(rf_mae))
        print("  - Insight: Neither the sequential data/LSTM approach nor the rolled features/RF approach")
        print("    provided a strong advantage for predicting the exact mood value in this run.")
        # Compare R2 as tie-breaker
        if abs(lstm_r2 - rf_r2) < 0.001: print(f"    R-squared values are also similar (~{rf_r2:.4f}).")
        elif lstm_r2 > rf_r2: print(f"    However, LSTM explains slightly more variance (R2: {lstm_r2:.4f} vs {rf_r2:.4f}).")
        else: print(f"    However, Random Forest explains slightly more variance (R2: {rf_r2:.4f} vs {lstm_r2:.4f}).")

    elif lstm_mae < rf_mae:
        print(f"  - LSTM Regressor is better (Lower MAE: {lstm_mae:.4f} vs {rf_mae:.4f}).")
        print("  - Insight: Capturing time-based patterns with LSTM on sequential data")
        print("    likely provided more predictive power for the continuous mood value")
        print("    than the rolled-up features used by Random Forest in this run.")
        if lstm_r2 > rf_r2: print(f"    The higher R-squared ({lstm_r2:.4f} vs {rf_r2:.4f}) further supports LSTM's advantage.")

    else: # rf_mae < lstm_mae
        print(f"  - Random Forest Regressor is better (Lower MAE: {rf_mae:.4f} vs {lstm_mae:.4f}).")
        print("  - Insight: The engineered features from rolling data combined with RF's")
        print("    ability to model interactions appears more effective than the LSTM's")
        print("    sequential approach for this regression task in this run.")
        if rf_r2 > lstm_r2: print(f"    The higher R-squared ({rf_r2:.4f} vs {lstm_r2:.4f}) further supports RF's advantage.")

elif valid_lstm:
     print("  - Only LSTM Regressor results are available for comparison.")
     print(f"    MAE: {lstm_mae:.4f}, MSE: {lstm_mse:.4f}, R2: {lstm_r2:.4f}")
elif valid_rf:
     print("  - Only Random Forest Regressor results are available for comparison.")
     print(f"    MAE: {rf_mae:.4f}, MSE: {rf_mse:.4f}, R2: {rf_r2:.4f}")
else:
    print("  - Comparison not possible due to errors in both models.")

print("\n" + "="*60)
print("               End of Comparison")
print("="*60)