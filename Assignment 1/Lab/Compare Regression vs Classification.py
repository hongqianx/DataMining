import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold, cross_val_score
import logging
import keras_tuner as kt
import optuna
import os
import time

# --- Configuration ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ModelComparisonLogger")

try:
    physical_devices_gpu = tf.config.list_physical_devices('GPU')
    if physical_devices_gpu:
        tf.config.set_visible_devices([], 'GPU')
        logger.info("TensorFlow GPU usage disabled.")
    else:
        logger.info("No GPU found or TensorFlow GPU support not available. Using CPU.")
except Exception as e:
    logger.warning(f"Could not configure TensorFlow devices: {e}")

# --- File Paths ---
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
results = {}

# --- Classification Helper Function ---
def split_mood_segments(mood):
    """Splits continuous mood into discrete segments."""
    if pd.isna(mood):
        return -1
    if mood <= 4: return 0
    elif mood <= 6: return 1
    elif mood <= 8: return 2
    else: return 3

# --- Task 4: Regression ---
logger.info("--- Starting Task 4: Regression ---")

# == 4.1 Random Forest Regressor ==
start_time = time.time()
logger.info("Running Random Forest Regressor (Task 4)...")
try:
    df_reg_rf = pd.read_csv(rolling_data_path)
    X_reg_rf = df_reg_rf.drop(columns=[prediction_col], errors='ignore').fillna(0)
    if prediction_col not in df_reg_rf.columns: raise ValueError(f"Target column '{prediction_col}' not found in {rolling_data_path}")
    y_reg_rf = df_reg_rf[prediction_col].fillna(df_reg_rf[prediction_col].mean())

    missing_cols_train = set(feature_cols) - set(X_reg_rf.columns)
    if missing_cols_train:
        logger.warning(f"Missing columns in rolling data for RF Regressor: {missing_cols_train}. Filling with 0.")
        for col in missing_cols_train: X_reg_rf[col] = 0
    extra_cols_train = set(X_reg_rf.columns) - set(feature_cols)
    if extra_cols_train:
         logger.warning(f"Extra columns found in rolling data for RF Regressor: {extra_cols_train}. Dropping them.")
         X_reg_rf = X_reg_rf.drop(columns=list(extra_cols_train))
    X_reg_rf = X_reg_rf[[col for col in feature_cols if col in X_reg_rf.columns]]

    X_train_reg_rf, X_test_reg_rf, y_train_reg_rf, y_test_reg_rf = train_test_split(X_reg_rf, y_reg_rf, test_size=test_size, random_state=random_seed)

    model_reg_rf = RandomForestRegressor(random_state=random_seed)
    search_space_reg_rf = {'n_estimators': [50, 100, 150], 'max_depth': [5, 10, 15, None], 'min_samples_split': [2, 4], 'min_samples_leaf': [1, 3], 'max_features': ["sqrt", "log2"]}
    tuned_model_reg_rf = GridSearchCV(model_reg_rf, search_space_reg_rf, cv=3, scoring='r2', n_jobs=-1, verbose=0)
    logger.info("Fitting RF Regressor GridSearchCV...")
    fitted_model_reg_rf = tuned_model_reg_rf.fit(X_train_reg_rf, y_train_reg_rf)
    best_params_reg_rf = fitted_model_reg_rf.best_params_
    logger.info(f"RF Regressor Best Hyperparameters: {best_params_reg_rf}")

    final_model_reg_rf = fitted_model_reg_rf.best_estimator_
    y_pred_reg_rf = final_model_reg_rf.predict(X_test_reg_rf)
    mse_reg_rf = mean_squared_error(y_test_reg_rf, y_pred_reg_rf)
    r2_reg_rf = r2_score(y_test_reg_rf, y_pred_reg_rf)
    mae_reg_rf = mean_absolute_error(y_test_reg_rf, y_pred_reg_rf)

    logger.info("Converting RF Regressor predictions to labels for accuracy check...")
    y_pred_labels_reg_rf = np.array([split_mood_segments(pred) for pred in y_pred_reg_rf])
    y_test_labels_reg_rf = np.array([split_mood_segments(true_val) for true_val in y_test_reg_rf])
    valid_indices_reg_rf_acc = (y_pred_labels_reg_rf != -1) & (y_test_labels_reg_rf != -1)
    if np.sum(~valid_indices_reg_rf_acc) > 0: logger.warning(f"Filtered out {np.sum(~valid_indices_reg_rf_acc)} samples for RF Regressor accuracy calculation due to invalid split labels.")
    acc_reg_rf = accuracy_score(y_test_labels_reg_rf[valid_indices_reg_rf_acc], y_pred_labels_reg_rf[valid_indices_reg_rf_acc])
    f1_reg_rf_as_cls = f1_score(y_test_labels_reg_rf[valid_indices_reg_rf_acc], y_pred_labels_reg_rf[valid_indices_reg_rf_acc], average='weighted', zero_division=0)
    logger.info(f"RF Regressor 'as classification' Accuracy: {acc_reg_rf:.4f}, F1: {f1_reg_rf_as_cls:.4f}")

    results['RF_Regressor'] = {
        'Best Params': best_params_reg_rf, 'MSE': mse_reg_rf, 'R2': r2_reg_rf, 'MAE': mae_reg_rf,
        'Accuracy (converted)': acc_reg_rf, 'F1 (converted)': f1_reg_rf_as_cls,
        'Feature Importances': pd.Series(final_model_reg_rf.feature_importances_, index=X_reg_rf.columns)
    }
    logger.info(f"RF Regressor Metrics - MSE: {mse_reg_rf:.4f}, R2: {r2_reg_rf:.4f}, MAE: {mae_reg_rf:.4f}, Accuracy (converted): {acc_reg_rf:.4f}")

    plt.figure(figsize=(8, 6))
    results['RF_Regressor']['Feature Importances'].nlargest(10).plot(kind='barh')
    plt.title("RF Regressor: Top 10 Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("rf_regressor_feature_importance.png")
    logger.info("Saved RF Regressor feature importance plot.")
    plt.close()

except FileNotFoundError:
    logger.error(f"Skipping RF Regressor: Data file not found at {rolling_data_path}")
    results['RF_Regressor'] = {'Error': 'Data file not found'}
except Exception as e:
    logger.error(f"Error during RF Regressor execution: {e}", exc_info=True)
    results['RF_Regressor'] = {'Error': str(e)}
end_time = time.time()
logger.info(f"RF Regressor (Task 4) took {end_time - start_time:.2f} seconds.")


# == 4.2 LSTM Regressor ==
start_time = time.time()
logger.info("Running LSTM Regressor (Task 4)...")
try:
    df_reg_lstm = pd.read_csv(interp_data_path)
    scaler_reg_lstm = StandardScaler()
    cols_to_scale_reg = [col for col in feature_cols if col in df_reg_lstm.columns]
    available_feature_cols_lstm_reg = cols_to_scale_reg
    if len(available_feature_cols_lstm_reg) < len(feature_cols): logger.warning(f"Using subset of features for LSTM Regressor due to missing columns in {interp_data_path}: {available_feature_cols_lstm_reg}")
    if not available_feature_cols_lstm_reg: raise ValueError("No feature columns found for LSTM Regressor.")
    df_reg_lstm[available_feature_cols_lstm_reg] = scaler_reg_lstm.fit_transform(df_reg_lstm[available_feature_cols_lstm_reg])
    df_reg_lstm = df_reg_lstm.sort_values(by=['id', 'time_bin'])

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

    def build_model_reg(hp):
        model = Sequential([
            LSTM(units=hp.Int('units', 32, 96, step=32), return_sequences=False, input_shape=(X_train_reg_lstm.shape[1], X_train_reg_lstm.shape[2])),
            Dropout(hp.Choice('dropout_rate', [0.1, 0.3, 0.5])),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=hp.Choice('learning_rate', [1e-3, 1e-4])), loss='mean_absolute_error', metrics=['mae'])
        return model
    tuner_reg_lstm = kt.RandomSearch(build_model_reg, objective='val_mae', max_trials=10, executions_per_trial=1, directory='kt_reg_dir', project_name='lstm_regression', overwrite=True)
    logger.info("Searching LSTM Regressor hyperparameters...")
    tuner_reg_lstm.search(X_train_reg_lstm, y_train_reg_lstm, epochs=10, batch_size=64, validation_data=(X_test_reg_lstm, y_test_reg_lstm), verbose=0, callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=3)])
    best_hp_reg_lstm = tuner_reg_lstm.get_best_hyperparameters(num_trials=1)[0]
    logger.info(f"LSTM Regressor Best Hyperparameters: {best_hp_reg_lstm.values}")

    best_model_reg_lstm = tuner_reg_lstm.get_best_models(num_models=1)[0]
    loss_reg_lstm, mae_reg_lstm = best_model_reg_lstm.evaluate(X_test_reg_lstm, y_test_reg_lstm, verbose=0)
    y_pred_reg_lstm = best_model_reg_lstm.predict(X_test_reg_lstm).flatten()
    mse_reg_lstm = mean_squared_error(y_test_reg_lstm, y_pred_reg_lstm)
    r2_reg_lstm = r2_score(y_test_reg_lstm, y_pred_reg_lstm)

    logger.info("Converting LSTM Regressor predictions to labels for accuracy check...")
    y_pred_labels_reg_lstm = np.array([split_mood_segments(pred) for pred in y_pred_reg_lstm])
    y_test_labels_reg_lstm = np.array([split_mood_segments(true_val) for true_val in y_test_reg_lstm])
    valid_indices_reg_lstm_acc = (y_pred_labels_reg_lstm != -1) & (y_test_labels_reg_lstm != -1)
    if np.sum(~valid_indices_reg_lstm_acc) > 0: logger.warning(f"Filtered out {np.sum(~valid_indices_reg_lstm_acc)} samples for LSTM Regressor accuracy calculation due to invalid split labels.")
    acc_reg_lstm = accuracy_score(y_test_labels_reg_lstm[valid_indices_reg_lstm_acc], y_pred_labels_reg_lstm[valid_indices_reg_lstm_acc])
    f1_reg_lstm_as_cls = f1_score(y_test_labels_reg_lstm[valid_indices_reg_lstm_acc], y_pred_labels_reg_lstm[valid_indices_reg_lstm_acc], average='weighted', zero_division=0)
    logger.info(f"LSTM Regressor 'as classification' Accuracy: {acc_reg_lstm:.4f}, F1: {f1_reg_lstm_as_cls:.4f}")

    results['LSTM_Regressor'] = {
        'Best Params': best_hp_reg_lstm.values, 'Test Loss (MAE)': loss_reg_lstm, 'Test MAE': mae_reg_lstm,
        'Test MSE': mse_reg_lstm, 'Test R2': r2_reg_lstm,
        'Accuracy (converted)': acc_reg_lstm, 'F1 (converted)': f1_reg_lstm_as_cls
    }
    logger.info(f"LSTM Regressor Metrics - Test Loss (MAE): {loss_reg_lstm:.4f}, Test MAE: {mae_reg_lstm:.4f}, Test MSE: {mse_reg_lstm:.4f}, Test R2: {r2_reg_lstm:.4f}, Accuracy (converted): {acc_reg_lstm:.4f}")

except FileNotFoundError:
    logger.error(f"Skipping LSTM Regressor: Data file not found at {interp_data_path}")
    results['LSTM_Regressor'] = {'Error': 'Data file not found'}
except Exception as e:
    logger.error(f"Error during LSTM Regressor execution: {e}", exc_info=True)
    results['LSTM_Regressor'] = {'Error': str(e)}
end_time = time.time()
logger.info(f"LSTM Regressor (Task 4) took {end_time - start_time:.2f} seconds.")


# --- Task 2A: Classification ---
logger.info("--- Starting Task 2A: Classification ---")

# == 2A.1 Random Forest Classifier ==
start_time = time.time()
logger.info("Running Random Forest Classifier (Task 2A)...")
try:
    df_cls_rf = pd.read_csv(rolling_data_path)
    X_cls_rf = df_cls_rf.drop(columns=[prediction_col], errors='ignore').fillna(0)
    if prediction_col not in df_cls_rf.columns: raise ValueError(f"Target column '{prediction_col}' not found in {rolling_data_path}")
    y_cls_rf_cont = df_cls_rf[prediction_col]
    y_cls_rf = y_cls_rf_cont.apply(split_mood_segments)

    original_count = len(y_cls_rf)
    valid_indices = y_cls_rf[y_cls_rf != -1].index
    X_cls_rf = X_cls_rf.loc[valid_indices]
    y_cls_rf = y_cls_rf.loc[valid_indices]
    if len(y_cls_rf) < original_count: logger.warning(f"Dropped {original_count - len(y_cls_rf)} samples for RF Classifier due to invalid/missing mood values.")
    if len(y_cls_rf) == 0: raise ValueError("No valid target data remains for RF Classifier after filtering.")

    missing_cols_train_cls = set(feature_cols) - set(X_cls_rf.columns)
    if missing_cols_train_cls:
        logger.warning(f"Missing columns in rolling data for RF Classifier: {missing_cols_train_cls}. Filling with 0.")
        for col in missing_cols_train_cls: X_cls_rf[col] = 0
    extra_cols_train_cls = set(X_cls_rf.columns) - set(feature_cols)
    if extra_cols_train_cls:
        logger.warning(f"Extra columns found in rolling data for RF Classifier: {extra_cols_train_cls}. Dropping them.")
        X_cls_rf = X_cls_rf.drop(columns=list(extra_cols_train_cls))
    X_cls_rf = X_cls_rf[[col for col in feature_cols if col in X_cls_rf.columns]]

    X_train_cls_rf, X_test_cls_rf, y_train_cls_rf, y_test_cls_rf = train_test_split(X_cls_rf, y_cls_rf, test_size=test_size, random_state=random_seed, stratify=y_cls_rf)

    logger.info("Running initial GridSearch for RF Classifier...")
    baseline_model_cls_rf = RandomForestClassifier(random_state=random_seed)
    search_space_cls_rf = {'criterion': ['gini', 'entropy'], 'n_estimators': [50, 100, 150], 'max_depth': [5, 10, 15, None], 'min_samples_split': [2, 4], 'min_samples_leaf': [1, 3], 'max_features': ["sqrt", "log2"]}
    cv_cls = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_seed)
    tuned_model_cls_rf = GridSearchCV(baseline_model_cls_rf, search_space_cls_rf, cv=cv_cls, scoring='accuracy', n_jobs=-1, verbose=0)
    fitted_model_cls_rf = tuned_model_cls_rf.fit(X_train_cls_rf, y_train_cls_rf)
    best_grid_params_cls_rf = fitted_model_cls_rf.best_params_
    logger.info(f"RF Classifier Initial GridSearch Best Params: {best_grid_params_cls_rf}")

    def objective_rf_cls(trial):
        params = {
            "criterion": trial.suggest_categorical("criterion", [best_grid_params_cls_rf.get('criterion', 'gini')]),
            "n_estimators": trial.suggest_int("n_estimators", max(10, best_grid_params_cls_rf.get('n_estimators', 100) - 25), best_grid_params_cls_rf.get('n_estimators', 100) + 25),
            "max_depth": trial.suggest_int("max_depth", max(2, best_grid_params_cls_rf.get('max_depth', 10) - 5 if best_grid_params_cls_rf.get('max_depth') is not None else 5), (best_grid_params_cls_rf.get('max_depth', 10) + 5) if best_grid_params_cls_rf.get('max_depth') is not None else 20),
            "min_samples_split": trial.suggest_int("min_samples_split", max(2, best_grid_params_cls_rf.get('min_samples_split', 2) - 1), best_grid_params_cls_rf.get('min_samples_split', 2) + 2),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", max(1, best_grid_params_cls_rf.get('min_samples_leaf', 1) - 1), best_grid_params_cls_rf.get('min_samples_leaf', 1) + 2),
            "max_features": trial.suggest_categorical("max_features", [best_grid_params_cls_rf.get('max_features', 'sqrt')]),
            "random_state": random_seed, "n_jobs": -1, "class_weight": trial.suggest_categorical("class_weight", ['balanced', None])}
        if best_grid_params_cls_rf.get('max_depth') is None and "max_depth" in params: params["max_depth"] = trial.suggest_int("max_depth", 15, 30)
        clf = RandomForestClassifier(**params)
        score = cross_val_score(clf, X_train_cls_rf, y_train_cls_rf, cv=cv_cls, scoring="accuracy", n_jobs=-1)
        return score.mean()
    logger.info("Running Optuna tuning for RF Classifier...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study_rf_cls = optuna.create_study(direction="maximize")
    study_rf_cls.optimize(objective_rf_cls, n_trials=20)
    optuna_best_params_cls_rf = study_rf_cls.best_params
    logger.info(f"RF Classifier Optuna Best Params: {optuna_best_params_cls_rf}")
    logger.info(f"RF Classifier Optuna Best Score (CV Accuracy): {study_rf_cls.best_value:.4f}")

    final_model_cls_rf = RandomForestClassifier(**optuna_best_params_cls_rf, random_state=random_seed, n_jobs=-1)
    final_model_cls_rf.fit(X_train_cls_rf, y_train_cls_rf)
    y_pred_cls_rf = final_model_cls_rf.predict(X_test_cls_rf)
    f1_cls_rf = f1_score(y_test_cls_rf, y_pred_cls_rf, average='weighted', zero_division=0)
    acc_cls_rf = accuracy_score(y_test_cls_rf, y_pred_cls_rf)
    r2_cls_rf = r2_score(y_test_cls_rf, y_pred_cls_rf)
    mse_cls_rf = mean_squared_error(y_test_cls_rf, y_pred_cls_rf)

    results['RF_Classifier'] = {
        'Best Params (Optuna)': optuna_best_params_cls_rf, 'F1 (weighted)': f1_cls_rf, 'Accuracy': acc_cls_rf,
        'R2': r2_cls_rf, 'MSE': mse_cls_rf,
        'Feature Importances': pd.Series(final_model_cls_rf.feature_importances_, index=X_cls_rf.columns)
    }
    logger.info(f"RF Classifier Metrics - F1: {f1_cls_rf:.4f}, Accuracy: {acc_cls_rf:.4f}, R2: {r2_cls_rf:.4f}, MSE: {mse_cls_rf:.4f}")

    plt.figure(figsize=(8, 6))
    results['RF_Classifier']['Feature Importances'].nlargest(10).plot(kind='barh')
    plt.title("RF Classifier: Top 10 Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("rf_classifier_feature_importance.png")
    logger.info("Saved RF Classifier feature importance plot.")
    plt.close()

except FileNotFoundError:
    logger.error(f"Skipping RF Classifier: Data file not found at {rolling_data_path}")
    results['RF_Classifier'] = {'Error': 'Data file not found'}
except Exception as e:
    logger.error(f"Error during RF Classifier execution: {e}", exc_info=True)
    results['RF_Classifier'] = {'Error': str(e)}
end_time = time.time()
logger.info(f"RF Classifier (Task 2A) took {end_time - start_time:.2f} seconds.")


# == 2A.2 LSTM Classifier ==
start_time = time.time()
logger.info("Running LSTM Classifier (Task 2A)...")
try:
    df_cls_lstm = pd.read_csv(interp_data_path)
    scaler_cls_lstm = StandardScaler()
    cols_to_scale_cls = [col for col in feature_cols if col in df_cls_lstm.columns]
    available_feature_cols_lstm_cls = cols_to_scale_cls
    if len(available_feature_cols_lstm_cls) < len(feature_cols): logger.warning(f"Using subset of features for LSTM Classifier due to missing columns in {interp_data_path}: {available_feature_cols_lstm_cls}")
    if not available_feature_cols_lstm_cls: raise ValueError("No feature columns found for LSTM Classifier.")
    df_cls_lstm[available_feature_cols_lstm_cls] = scaler_cls_lstm.fit_transform(df_cls_lstm[available_feature_cols_lstm_cls])
    df_cls_lstm = df_cls_lstm.sort_values(by=['id', 'time_bin'])

    seq_length_cls = 6
    X_seq_cls_lstm, y_seq_cls_lstm_cont = [], []
    if prediction_col not in df_cls_lstm.columns: raise ValueError(f"Target column '{prediction_col}' not found in {interp_data_path}")
    grouped_cls = df_cls_lstm.groupby('id')
    for group_id, group in grouped_cls:
        if len(group) >= seq_length_cls:
            for i in range(seq_length_cls, len(group)):
                X_seq_cls_lstm.append(group[available_feature_cols_lstm_cls].iloc[i - seq_length_cls:i].values)
                y_seq_cls_lstm_cont.append(group[prediction_col].iloc[i])
        else: logger.warning(f"Skipping id {group_id} for LSTM Classifier sequencing: too few records ({len(group)} < {seq_length_cls})")
    if not X_seq_cls_lstm: raise ValueError("No sequences created for LSTM Classifier. Check data, grouping, and seq_length.")
    X_seq_cls_lstm = np.array(X_seq_cls_lstm)
    y_seq_cls_lstm = np.array([split_mood_segments(mood) for mood in y_seq_cls_lstm_cont])

    valid_indices_cls = (y_seq_cls_lstm != -1)
    original_count_cls = len(y_seq_cls_lstm)
    X_seq_cls_lstm = X_seq_cls_lstm[valid_indices_cls]
    y_seq_cls_lstm = y_seq_cls_lstm[valid_indices_cls]
    if len(y_seq_cls_lstm) < original_count_cls: logger.warning(f"Dropped {original_count_cls - len(y_seq_cls_lstm)} sequences for LSTM classifier due to invalid/missing mood values.")
    if len(y_seq_cls_lstm) == 0: raise ValueError("No valid target data remains for LSTM Classifier after filtering.")
    if X_seq_cls_lstm.shape[1] != seq_length_cls or X_seq_cls_lstm.shape[2] != len(available_feature_cols_lstm_cls): raise ValueError(f"Unexpected sequence shape for LSTM Classifier: {X_seq_cls_lstm.shape}. Expected ({len(X_seq_cls_lstm)}, {seq_length_cls}, {len(available_feature_cols_lstm_cls)})")
    num_classes = len(np.unique(y_seq_cls_lstm))
    logger.info(f"Number of classes for LSTM Classifier: {num_classes}")
    if num_classes <= 1: raise ValueError("LSTM Classifier requires more than one class after processing.")
    X_train_cls_lstm, X_test_cls_lstm, y_train_cls_lstm, y_test_cls_lstm = train_test_split(X_seq_cls_lstm, y_seq_cls_lstm, test_size=test_size, random_state=random_seed, stratify=y_seq_cls_lstm)

    def build_model_cls(hp):
        model = Sequential([
            LSTM(units=hp.Int('units', 32, 96, step=32), return_sequences=False, input_shape=(X_train_cls_lstm.shape[1], X_train_cls_lstm.shape[2])),
            Dropout(hp.Choice('dropout_rate', [0.1, 0.3, 0.5])),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer=Adam(learning_rate=hp.Choice('learning_rate', [1e-3, 1e-4])), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    tuner_cls_lstm = kt.RandomSearch(build_model_cls, objective='val_accuracy', max_trials=10, executions_per_trial=1, directory='kt_cls_dir', project_name='lstm_classification', overwrite=True)
    logger.info("Searching LSTM Classifier hyperparameters (Keras Tuner)...")
    tuner_cls_lstm.search(X_train_cls_lstm, y_train_cls_lstm, epochs=10, batch_size=64, validation_data=(X_test_cls_lstm, y_test_cls_lstm), verbose=0, callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=3)])
    best_hp_cls_lstm_kt = tuner_cls_lstm.get_best_hyperparameters(num_trials=1)[0]
    logger.info(f"LSTM Classifier Keras Tuner Best Hyperparameters: {best_hp_cls_lstm_kt.values}")
    best_model_cls_lstm_kt = tuner_cls_lstm.get_best_models(num_models=1)[0]
    kt_loss, kt_acc = best_model_cls_lstm_kt.evaluate(X_test_cls_lstm, y_test_cls_lstm, verbose=0)
    logger.info(f"LSTM Classifier Keras Tuner Best Model Eval - Loss: {kt_loss:.4f}, Accuracy: {kt_acc:.4f}")

    def objective_lstm_cls(trial):
        units = trial.suggest_int('units', max(16, best_hp_cls_lstm_kt.get('units') - 16), best_hp_cls_lstm_kt.get('units') + 16)
        dropout_rate = trial.suggest_float('dropout_rate', max(0.0, best_hp_cls_lstm_kt.get('dropout_rate') - 0.1), min(0.7, best_hp_cls_lstm_kt.get('dropout_rate') + 0.1), step=0.05)
        learning_rate = trial.suggest_float('learning_rate', best_hp_cls_lstm_kt.get('learning_rate') * 0.1, best_hp_cls_lstm_kt.get('learning_rate') * 10, log=True)
        model = Sequential([
            LSTM(units=units, return_sequences=False, input_shape=(X_train_cls_lstm.shape[1], X_train_cls_lstm.shape[2])),
            Dropout(dropout_rate),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(X_train_cls_lstm, y_train_cls_lstm, epochs=5, batch_size=64, validation_split=0.2, verbose=0, callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=2)])
        val_accuracy = max(history.history.get('val_accuracy', [0]))
        if not isinstance(val_accuracy, (float, int)): val_accuracy = 0.0
        return val_accuracy
    logger.info("Running Optuna tuning for LSTM Classifier...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study_lstm_cls = optuna.create_study(direction='maximize')
    study_lstm_cls.optimize(objective_lstm_cls, n_trials=20)
    optuna_best_params_cls_lstm = study_lstm_cls.best_params
    logger.info(f"LSTM Classifier Optuna Best Hyperparameters: {optuna_best_params_cls_lstm}")
    logger.info(f"LSTM Classifier Optuna Best Score (Approx Val Accuracy): {study_lstm_cls.best_value:.4f}")

    final_model_cls_lstm = Sequential([
        LSTM(units=optuna_best_params_cls_lstm['units'], return_sequences=False, input_shape=(X_train_cls_lstm.shape[1], X_train_cls_lstm.shape[2])),
        Dropout(optuna_best_params_cls_lstm['dropout_rate']),
        Dense(num_classes, activation='softmax')
    ])
    final_model_cls_lstm.compile(optimizer=Adam(learning_rate=optuna_best_params_cls_lstm['learning_rate']), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    logger.info("Training final LSTM Classifier model with Optuna parameters...")
    final_model_cls_lstm.fit(X_train_cls_lstm, y_train_cls_lstm, epochs=20, batch_size=64, validation_data=(X_test_cls_lstm, y_test_cls_lstm), verbose=0, callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=5)])

    loss_cls_lstm, acc_cls_lstm = final_model_cls_lstm.evaluate(X_test_cls_lstm, y_test_cls_lstm, verbose=0)
    predictions_cls_lstm = final_model_cls_lstm.predict(X_test_cls_lstm)
    y_pred_cls_lstm = np.argmax(predictions_cls_lstm, axis=1)
    f1_cls_lstm = f1_score(y_test_cls_lstm, y_pred_cls_lstm, average='weighted', zero_division=0)

    results['LSTM_Classifier'] = {
        'Best Params (Optuna)': optuna_best_params_cls_lstm, 'Test Loss': loss_cls_lstm,
        'Test Accuracy': acc_cls_lstm, 'Test F1 (weighted)': f1_cls_lstm
    }
    logger.info(f"LSTM Classifier Metrics - Test Loss: {loss_cls_lstm:.4f}, Accuracy: {acc_cls_lstm:.4f}, F1: {f1_cls_lstm:.4f}")

except FileNotFoundError:
    logger.error(f"Skipping LSTM Classifier: Data file not found at {interp_data_path}")
    results['LSTM_Classifier'] = {'Error': 'Data file not found'}
except Exception as e:
    logger.error(f"Error during LSTM Classifier execution: {e}", exc_info=True)
    results['LSTM_Classifier'] = {'Error': str(e)}
end_time = time.time()
logger.info(f"LSTM Classifier (Task 2A) took {end_time - start_time:.2f} seconds.")


# --- Final Accuracy Comparison ---
logger.info("--- Final Accuracy Comparison for Mood Label Prediction ---")

print("\n" + "="*60)
print("      Final Accuracy Comparison for Mood Label Prediction")
print("="*60 + "\n")

# Retrieve results safely using -1.0 as default for failed/missing runs
rf_cls_acc = results.get('RF_Classifier', {}).get('Accuracy', -1.0)
rf_reg_conv_acc = results.get('RF_Regressor', {}).get('Accuracy (converted)', -1.0)
lstm_cls_acc = results.get('LSTM_Classifier', {}).get('Test Accuracy', -1.0)
lstm_reg_conv_acc = results.get('LSTM_Regressor', {}).get('Accuracy (converted)', -1.0)

# Print individual results
print(f"Random Forest Classifier Accuracy:         {rf_cls_acc:.4f}" if rf_cls_acc != -1.0 else "Random Forest Classifier: Error/Not Run")
print(f"RF Regressor (converted) Accuracy:       {rf_reg_conv_acc:.4f}" if rf_reg_conv_acc != -1.0 else "RF Regressor (converted): Error/Not Run")
print(f"LSTM Classifier Accuracy:                {lstm_cls_acc:.4f}" if lstm_cls_acc != -1.0 else "LSTM Classifier: Error/Not Run")
print(f"LSTM Regressor (converted) Accuracy:     {lstm_reg_conv_acc:.4f}" if lstm_reg_conv_acc != -1.0 else "LSTM Regressor (converted): Error/Not Run")

# Determine the best approach based purely on accuracy
accuracies = {
    "Random Forest Classifier": rf_cls_acc,
    "RF Regressor (converted)": rf_reg_conv_acc,
    "LSTM Classifier": lstm_cls_acc,
    "LSTM Regressor (converted)": lstm_reg_conv_acc
}

# Filter out failed runs (-1.0 indicates failure or missing)
valid_accuracies = {k: v for k, v in accuracies.items() if v != -1.0}

if valid_accuracies:
    # Find the method with the highest accuracy
    best_method = max(valid_accuracies, key=valid_accuracies.get)
    best_accuracy = valid_accuracies[best_method]

    print("\n" + "-"*60)
    print(f"Conclusion for Mood Label Prediction:")
    print(f"  The best approach based purely on label accuracy is: {best_method}")
    print(f"  Highest Accuracy Achieved: {best_accuracy:.4f}")

    # Compare regression (converted) vs classification directly
    print("\nDirect Comparison (Converted Regressor vs Classifier):")
    # Random Forest Comparison
    if rf_cls_acc != -1.0 and rf_reg_conv_acc != -1.0:
        if abs(rf_reg_conv_acc - rf_cls_acc) < 0.001: # Consider them equal if very close
             print(f"  - Random Forest: Classifier and converted Regressor had similar accuracy (~{rf_cls_acc:.4f}).")
        elif rf_reg_conv_acc > rf_cls_acc:
            print(f"  - Random Forest: The Regressor (converted) was MORE accurate than the Classifier ({rf_reg_conv_acc:.4f} vs {rf_cls_acc:.4f}).")
        else: # rf_cls_acc > rf_reg_conv_acc
            print(f"  - Random Forest: The Classifier was MORE accurate than the Regressor (converted) ({rf_cls_acc:.4f} vs {rf_reg_conv_acc:.4f}).")
    else:
        print("  - Random Forest: Comparison not possible due to errors/missing results.")

    # LSTM Comparison
    if lstm_cls_acc != -1.0 and lstm_reg_conv_acc != -1.0:
         if abs(lstm_reg_conv_acc - lstm_cls_acc) < 0.001: # Consider them equal if very close
            print(f"  - LSTM: Classifier and converted Regressor had similar accuracy (~{lstm_cls_acc:.4f}).")
         elif lstm_reg_conv_acc > lstm_cls_acc:
            print(f"  - LSTM: The Regressor (converted) was MORE accurate than the Classifier ({lstm_reg_conv_acc:.4f} vs {lstm_cls_acc:.4f}).")
         else: # lstm_cls_acc > lstm_reg_conv_acc
            print(f"  - LSTM: The Classifier was MORE accurate than the Regressor (converted) ({lstm_cls_acc:.4f} vs {lstm_reg_conv_acc:.4f}).")
    else:
         print("  - LSTM: Comparison not possible due to errors/missing results.")

else:
    print("\n" + "-"*60)
    print("Conclusion: Could not determine the best method due to errors or no successful runs.")

print("\n" + "="*60)
print("               End of Summary")
print("="*60)