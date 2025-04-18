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
import time # Added to time execution

# --- Configuration ---

# Set up a basic logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ModelComparisonLogger")

# Disable GPU usage for TensorFlow (modify if GPU is desired and configured)
try:
    tf.config.set_visible_devices([], 'GPU')
    logger.info("TensorFlow GPU usage disabled.")
except Exception as e:
    logger.warning(f"Could not configure TensorFlow devices: {e}")

# --- File Paths ---
rolling_data_path = "../../input/df_rolling.csv"
interp_data_path = "../../input/df_interp_6hour.csv"

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
    if mood <= 4:
        return 0  # low
    elif mood <= 6:
        return 1  # neutral
    elif mood <= 8:
        return 2  # good
    else: # mood > 8
        return 3  # great

# --- Task 4: Regression ---
logger.info("--- Starting Task 4: Regression ---")

# == 4.1 Random Forest Regressor ==
start_time = time.time()
logger.info("Running Random Forest Regressor (Task 4)...")
try:
    df_reg_rf = pd.read_csv(rolling_data_path)

    # Prepare data
    X_reg_rf = df_reg_rf.drop(columns=[prediction_col]).fillna(0)
    y_reg_rf = df_reg_rf[prediction_col].fillna(df_reg_rf[prediction_col].mean())

    # Ensure feature columns match the defined list, handling potential discrepancies
    missing_cols_train = set(feature_cols) - set(X_reg_rf.columns)
    if missing_cols_train:
        logger.warning(f"Missing columns in rolling data for RF Regressor: {missing_cols_train}. Filling with 0.")
        for col in missing_cols_train:
            X_reg_rf[col] = 0
    extra_cols_train = set(X_reg_rf.columns) - set(feature_cols)
    if extra_cols_train:
         logger.warning(f"Extra columns found in rolling data for RF Regressor: {extra_cols_train}. Dropping them.")
         X_reg_rf = X_reg_rf.drop(columns=list(extra_cols_train))
    X_reg_rf = X_reg_rf[feature_cols] # Ensure correct order and columns

    X_train_reg_rf, X_test_reg_rf, y_train_reg_rf, y_test_reg_rf = train_test_split(
        X_reg_rf, y_reg_rf, test_size=test_size, random_state=random_seed
    )

    # Hyperparameter Tuning (GridSearch)
    model_reg_rf = RandomForestRegressor(random_state=random_seed)
    search_space_reg_rf = {
        'n_estimators': [50, 100, 150],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 4],
        'min_samples_leaf': [1, 3],
        'max_features': ["sqrt", "log2"]
    }
    tuned_model_reg_rf = GridSearchCV(model_reg_rf, search_space_reg_rf, cv=3, scoring='r2', n_jobs=-1, verbose=0)
    logger.info("Fitting RF Regressor GridSearchCV...")
    fitted_model_reg_rf = tuned_model_reg_rf.fit(X_train_reg_rf, y_train_reg_rf)
    best_params_reg_rf = fitted_model_reg_rf.best_params_
    logger.info(f"RF Regressor Best Hyperparameters: {best_params_reg_rf}")

    # Evaluate
    final_model_reg_rf = fitted_model_reg_rf.best_estimator_
    y_pred_reg_rf = final_model_reg_rf.predict(X_test_reg_rf)
    mse_reg_rf = mean_squared_error(y_test_reg_rf, y_pred_reg_rf)
    r2_reg_rf = r2_score(y_test_reg_rf, y_pred_reg_rf)
    mae_reg_rf = mean_absolute_error(y_test_reg_rf, y_pred_reg_rf)

    # Store results
    results['RF_Regressor'] = {
        'Best Params': best_params_reg_rf,
        'MSE': mse_reg_rf,
        'R2': r2_reg_rf,
        'MAE': mae_reg_rf,
        'Feature Importances': pd.Series(final_model_reg_rf.feature_importances_, index=X_reg_rf.columns)
    }
    logger.info(f"RF Regressor Metrics - MSE: {mse_reg_rf:.4f}, R2: {r2_reg_rf:.4f}, MAE: {mae_reg_rf:.4f}")

    # Plot Feature Importance
    plt.figure(figsize=(8, 6))
    results['RF_Regressor']['Feature Importances'].nlargest(10).plot(kind='barh')
    plt.title("RF Regressor: Top 10 Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    # plt.show() # Uncomment to display plot interactively
    plt.savefig("rf_regressor_feature_importance.png")
    logger.info("Saved RF Regressor feature importance plot.")
    plt.close()

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

    # Prepare data
    scaler_reg_lstm = StandardScaler()
    # Ensure feature columns exist before scaling
    cols_to_scale_reg = [col for col in feature_cols if col in df_reg_lstm.columns]
    if len(cols_to_scale_reg) < len(feature_cols):
        logger.warning("Not all feature columns found in interp data for LSTM Regressor scaling.")

    if cols_to_scale_reg: # Proceed only if there are columns to scale
      df_reg_lstm[cols_to_scale_reg] = scaler_reg_lstm.fit_transform(df_reg_lstm[cols_to_scale_reg])

    df_reg_lstm = df_reg_lstm.sort_values(by=['id', 'time_bin'])

    # Create sequences
    seq_length_reg = 6
    X_seq_reg_lstm = []
    y_seq_reg_lstm = []

    # Check if target column exists
    if prediction_col not in df_reg_lstm.columns:
        raise ValueError(f"Target column '{prediction_col}' not found in {interp_data_path}")

    df_reg_lstm[prediction_col] = df_reg_lstm[prediction_col].fillna(df_reg_lstm[prediction_col].mean())

    # Use only available feature columns for sequences
    available_feature_cols_lstm_reg = [col for col in feature_cols if col in df_reg_lstm.columns]
    if len(available_feature_cols_lstm_reg) < len(feature_cols):
       logger.warning(f"Using subset of features for LSTM Regressor due to missing columns: {available_feature_cols_lstm_reg}")

    if not available_feature_cols_lstm_reg:
        raise ValueError("No feature columns available for LSTM Regressor sequencing.")


    grouped = df_reg_lstm.groupby('id')
    for _, group in grouped:
        for i in range(seq_length_reg, len(group)):
            X_seq_reg_lstm.append(group[available_feature_cols_lstm_reg].iloc[i - seq_length_reg:i].values)
            y_seq_reg_lstm.append(group[prediction_col].iloc[i])


    if not X_seq_reg_lstm:
        raise ValueError("No sequences created for LSTM Regressor. Check data and seq_length.")

    X_seq_reg_lstm = np.array(X_seq_reg_lstm)
    y_seq_reg_lstm = np.array(y_seq_reg_lstm)

    # Check sequence shape
    if X_seq_reg_lstm.shape[1] != seq_length_reg or X_seq_reg_lstm.shape[2] != len(available_feature_cols_lstm_reg):
        raise ValueError(f"Unexpected sequence shape for LSTM Regressor: {X_seq_reg_lstm.shape}. Expected ({len(X_seq_reg_lstm)}, {seq_length_reg}, {len(available_feature_cols_lstm_reg)})")


    X_train_reg_lstm, X_test_reg_lstm, y_train_reg_lstm, y_test_reg_lstm = train_test_split(
        X_seq_reg_lstm, y_seq_reg_lstm, test_size=test_size, random_state=random_seed
    )

    # Hyperparameter Tuning (Keras Tuner)
    def build_model_reg(hp):
        model = Sequential()
        model.add(LSTM(
            units=hp.Int('units', min_value=32, max_value=96, step=32),
            return_sequences=False,
            # Use the actual shape from the prepared training data
            input_shape=(X_train_reg_lstm.shape[1], X_train_reg_lstm.shape[2])
        ))
        model.add(Dropout(hp.Choice('dropout_rate', [0.1, 0.3, 0.5])))
        model.add(Dense(1, activation='linear')) # Regression output
        model.compile(
            optimizer=Adam(learning_rate=hp.Choice('learning_rate', [1e-3, 1e-4])),
            loss='mean_absolute_error',
            metrics=['mae']
        )
        return model

    tuner_reg_lstm = kt.RandomSearch(
        build_model_reg,
        objective='val_mae', # Use validation MAE
        max_trials=10,
        executions_per_trial=1,
        directory='kt_reg_dir',
        project_name='lstm_regression',
        overwrite=True
    )

    logger.info("Searching LSTM Regressor hyperparameters...")
    tuner_reg_lstm.search(X_train_reg_lstm, y_train_reg_lstm, epochs=10, batch_size=64,
                        validation_data=(X_test_reg_lstm, y_test_reg_lstm), verbose=0,
                        callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=3)])

    best_hp_reg_lstm = tuner_reg_lstm.get_best_hyperparameters(num_trials=1)[0]
    logger.info(f"LSTM Regressor Best Hyperparameters: {best_hp_reg_lstm.values}")

    # Evaluate Best Model
    best_model_reg_lstm = tuner_reg_lstm.get_best_models(num_models=1)[0]
    loss_reg_lstm, mae_reg_lstm = best_model_reg_lstm.evaluate(X_test_reg_lstm, y_test_reg_lstm, verbose=0)
    y_pred_reg_lstm = best_model_reg_lstm.predict(X_test_reg_lstm).flatten() # Flatten predictions
    mse_reg_lstm = mean_squared_error(y_test_reg_lstm, y_pred_reg_lstm) # Calculate MSE
    r2_reg_lstm = r2_score(y_test_reg_lstm, y_pred_reg_lstm) # Calculate R2

    # Store results
    results['LSTM_Regressor'] = {
        'Best Params': best_hp_reg_lstm.values,
        'Test Loss (MAE)': loss_reg_lstm,
        'Test MAE': mae_reg_lstm,
        'Test MSE': mse_reg_lstm,
        'Test R2': r2_reg_lstm,
    }
    logger.info(f"LSTM Regressor Metrics - Test Loss (MAE): {loss_reg_lstm:.4f}, Test MAE: {mae_reg_lstm:.4f}, Test MSE: {mse_reg_lstm:.4f}, Test R2: {r2_reg_lstm:.4f}")

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

    # Prepare data
    X_cls_rf = df_cls_rf.drop(columns=[prediction_col]).fillna(0)
    # Apply classification split function
    y_cls_rf = df_cls_rf[prediction_col].apply(split_mood_segments)

    original_count = len(y_cls_rf)
    valid_indices = y_cls_rf[y_cls_rf != -1].index
    X_cls_rf = X_cls_rf.loc[valid_indices]
    y_cls_rf = y_cls_rf.loc[valid_indices]
    if len(y_cls_rf) < original_count:
        logger.warning(f"Dropped {original_count - len(y_cls_rf)} samples due to invalid mood for classification.")

    # Ensure feature columns match the defined list
    missing_cols_train_cls = set(feature_cols) - set(X_cls_rf.columns)
    if missing_cols_train_cls:
        logger.warning(f"Missing columns in rolling data for RF Classifier: {missing_cols_train_cls}. Filling with 0.")
        for col in missing_cols_train_cls:
            X_cls_rf[col] = 0
    extra_cols_train_cls = set(X_cls_rf.columns) - set(feature_cols)
    if extra_cols_train_cls:
        logger.warning(f"Extra columns found in rolling data for RF Classifier: {extra_cols_train_cls}. Dropping them.")
        X_cls_rf = X_cls_rf.drop(columns=list(extra_cols_train_cls))
    X_cls_rf = X_cls_rf[feature_cols] # Ensure correct order and columns


    X_train_cls_rf, X_test_cls_rf, y_train_cls_rf, y_test_cls_rf = train_test_split(
        X_cls_rf, y_cls_rf, test_size=test_size, random_state=random_seed, stratify=y_cls_rf # Stratify for classification
    )

    # Hyperparameter Tuning (GridSearch + Optuna)
    # 1. GridSearch (coarse)
    logger.info("Running initial GridSearch for RF Classifier...")
    baseline_model_cls_rf = RandomForestClassifier(random_state=random_seed)
    search_space_cls_rf = {
        'criterion': ['gini', 'entropy'],
        'n_estimators': [50, 100, 150], # Reduced range
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 4],
        'min_samples_leaf': [1, 3],
        'max_features': ["sqrt", "log2"]
    }
    cv_cls = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_seed)
    tuned_model_cls_rf = GridSearchCV(baseline_model_cls_rf, search_space_cls_rf, cv=cv_cls, scoring='accuracy', n_jobs=-1, verbose=0)
    fitted_model_cls_rf = tuned_model_cls_rf.fit(X_train_cls_rf, y_train_cls_rf)
    best_grid_params_cls_rf = fitted_model_cls_rf.best_params_
    logger.info(f"RF Classifier Initial GridSearch Best Params: {best_grid_params_cls_rf}")

    # 2. Optuna (fine-tuning) - Define objective function inside the try block
    def objective_rf_cls(trial):
        # Define search space relative to GridSearch results
        params = {
            "criterion": trial.suggest_categorical("criterion", [best_grid_params_cls_rf.get('criterion', 'gini')]),
            "n_estimators": trial.suggest_int("n_estimators",
                                              max(10, best_grid_params_cls_rf.get('n_estimators', 100) - 25),
                                              best_grid_params_cls_rf.get('n_estimators', 100) + 25),
            "max_depth": trial.suggest_int("max_depth",
                                           max(2, best_grid_params_cls_rf.get('max_depth', 10) - 5 if best_grid_params_cls_rf.get('max_depth') is not None else 5),
                                           (best_grid_params_cls_rf.get('max_depth', 10) + 5) if best_grid_params_cls_rf.get('max_depth') is not None else 20),
            "min_samples_split": trial.suggest_int("min_samples_split",
                                                  max(2, best_grid_params_cls_rf.get('min_samples_split', 2) - 1),
                                                  best_grid_params_cls_rf.get('min_samples_split', 2) + 2),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf",
                                                 max(1, best_grid_params_cls_rf.get('min_samples_leaf', 1) - 1),
                                                 best_grid_params_cls_rf.get('min_samples_leaf', 1) + 2),
            "max_features": trial.suggest_categorical("max_features", [best_grid_params_cls_rf.get('max_features', 'sqrt')]),
            "random_state": random_seed,
            "n_jobs": -1,
            "class_weight": trial.suggest_categorical("class_weight", ['balanced', None])
        }
        # Handle max_depth if None was best in grid search
        if best_grid_params_cls_rf.get('max_depth') is None and "max_depth" in params:
             params["max_depth"] = trial.suggest_int("max_depth", 15, 30)

        clf = RandomForestClassifier(**params)
        # Use cross_val_score for Optuna evaluation
        score = cross_val_score(clf, X_train_cls_rf, y_train_cls_rf, cv=cv_cls, scoring="accuracy", n_jobs=-1)
        return score.mean()

    logger.info("Running Optuna tuning for RF Classifier...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study_rf_cls = optuna.create_study(direction="maximize")
    study_rf_cls.optimize(objective_rf_cls, n_trials=20)
    optuna_best_params_cls_rf = study_rf_cls.best_params
    logger.info(f"RF Classifier Optuna Best Params: {optuna_best_params_cls_rf}")
    logger.info(f"RF Classifier Optuna Best Score (CV Accuracy): {study_rf_cls.best_value:.4f}")

    # Evaluate Final Model (using Optuna best params)
    final_model_cls_rf = RandomForestClassifier(**optuna_best_params_cls_rf, random_state=random_seed, n_jobs=-1)
    final_model_cls_rf.fit(X_train_cls_rf, y_train_cls_rf)
    y_pred_cls_rf = final_model_cls_rf.predict(X_test_cls_rf)

    f1_cls_rf = f1_score(y_test_cls_rf, y_pred_cls_rf, average='weighted')
    acc_cls_rf = accuracy_score(y_test_cls_rf, y_pred_cls_rf)
    # Calculate R2 and MSE for comparison context (though less ideal for classification)
    r2_cls_rf = r2_score(y_test_cls_rf, y_pred_cls_rf)
    mse_cls_rf = mean_squared_error(y_test_cls_rf, y_pred_cls_rf)

    # Store results
    results['RF_Classifier'] = {
        'Best Params (Optuna)': optuna_best_params_cls_rf,
        'F1 (weighted)': f1_cls_rf,
        'Accuracy': acc_cls_rf,
        'R2': r2_cls_rf,
        'MSE': mse_cls_rf,
        'Feature Importances': pd.Series(final_model_cls_rf.feature_importances_, index=X_cls_rf.columns)
    }
    logger.info(f"RF Classifier Metrics - F1: {f1_cls_rf:.4f}, Accuracy: {acc_cls_rf:.4f}, R2: {r2_cls_rf:.4f}, MSE: {mse_cls_rf:.4f}")

    # Plot Feature Importance
    plt.figure(figsize=(8, 6))
    results['RF_Classifier']['Feature Importances'].nlargest(10).plot(kind='barh')
    plt.title("RF Classifier: Top 10 Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    # plt.show() # Uncomment to display plot interactively
    plt.savefig("rf_classifier_feature_importance.png")
    logger.info("Saved RF Classifier feature importance plot.")
    plt.close()

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

    # Prepare data
    scaler_cls_lstm = StandardScaler()
     # Ensure feature columns exist before scaling
    cols_to_scale_cls = [col for col in feature_cols if col in df_cls_lstm.columns]
    if len(cols_to_scale_cls) < len(feature_cols):
        logger.warning("Not all feature columns found in interp data for LSTM Classifier scaling.")

    if cols_to_scale_cls: # Proceed only if there are columns to scale
      df_cls_lstm[cols_to_scale_cls] = scaler_cls_lstm.fit_transform(df_cls_lstm[cols_to_scale_cls])

    df_cls_lstm = df_cls_lstm.sort_values(by=['id', 'time_bin'])

    # Create sequences
    seq_length_cls = 6
    X_seq_cls_lstm = []
    y_seq_cls_lstm_cont = [] # Continuous mood before splitting

     # Check if target column exists
    if prediction_col not in df_cls_lstm.columns:
        raise ValueError(f"Target column '{prediction_col}' not found in {interp_data_path}")

    # Use only available feature columns for sequences
    available_feature_cols_lstm_cls = [col for col in feature_cols if col in df_cls_lstm.columns]
    if len(available_feature_cols_lstm_cls) < len(feature_cols):
       logger.warning(f"Using subset of features for LSTM Classifier due to missing columns: {available_feature_cols_lstm_cls}")

    if not available_feature_cols_lstm_cls:
        raise ValueError("No feature columns available for LSTM Classifier sequencing.")

    grouped_cls = df_cls_lstm.groupby('id')
    for _, group in grouped_cls:
         for i in range(seq_length_cls, len(group)):
             X_seq_cls_lstm.append(group[available_feature_cols_lstm_cls].iloc[i - seq_length_cls:i].values)
             y_seq_cls_lstm_cont.append(group[prediction_col].iloc[i]) # Get continuous mood

    if not X_seq_cls_lstm:
        raise ValueError("No sequences created for LSTM Classifier. Check data and seq_length.")

    X_seq_cls_lstm = np.array(X_seq_cls_lstm)
    # Apply classification split function to the sequenced target
    y_seq_cls_lstm = np.array([split_mood_segments(mood) for mood in y_seq_cls_lstm_cont])

    # Handle potential -1s after sequencing and splitting
    valid_indices_cls = (y_seq_cls_lstm != -1)
    original_count_cls = len(y_seq_cls_lstm)
    X_seq_cls_lstm = X_seq_cls_lstm[valid_indices_cls]
    y_seq_cls_lstm = y_seq_cls_lstm[valid_indices_cls]
    if len(y_seq_cls_lstm) < original_count_cls:
         logger.warning(f"Dropped {original_count_cls - len(y_seq_cls_lstm)} sequences due to invalid mood for LSTM classification.")

    # Check sequence shape
    if X_seq_cls_lstm.shape[1] != seq_length_cls or X_seq_cls_lstm.shape[2] != len(available_feature_cols_lstm_cls):
        raise ValueError(f"Unexpected sequence shape for LSTM Classifier: {X_seq_cls_lstm.shape}. Expected ({len(X_seq_cls_lstm)}, {seq_length_cls}, {len(available_feature_cols_lstm_cls)})")

    # Determine number of classes
    num_classes = len(np.unique(y_seq_cls_lstm))
    logger.info(f"Number of classes for LSTM Classifier: {num_classes}")
    if num_classes <= 1:
      raise ValueError("LSTM Classifier requires more than one class after processing.")

    X_train_cls_lstm, X_test_cls_lstm, y_train_cls_lstm, y_test_cls_lstm = train_test_split(
        X_seq_cls_lstm, y_seq_cls_lstm, test_size=test_size, random_state=random_seed, stratify=y_seq_cls_lstm # Stratify
    )

    # Hyperparameter Tuning (Keras Tuner + Optuna)
    # 1. Keras Tuner (coarse)
    def build_model_cls(hp):
        model = Sequential()
        model.add(LSTM(
            units=hp.Int('units', min_value=32, max_value=96, step=32),
            return_sequences=False,
            input_shape=(X_train_cls_lstm.shape[1], X_train_cls_lstm.shape[2])
        ))
        model.add(Dropout(hp.Choice('dropout_rate', [0.1, 0.3, 0.5])))
        model.add(Dense(num_classes, activation='softmax')) # Classification output
        model.compile(
            optimizer=Adam(learning_rate=hp.Choice('learning_rate', [1e-3, 1e-4])),
            loss='sparse_categorical_crossentropy', # Use sparse version as y_train is integer labels
            metrics=['accuracy']
        )
        return model

    tuner_cls_lstm = kt.RandomSearch(
        build_model_cls,
        objective='val_accuracy', # Optimize for validation accuracy
        max_trials=10,
        executions_per_trial=1,
        directory='kt_cls_dir',
        project_name='lstm_classification',
        overwrite=True
    )

    logger.info("Searching LSTM Classifier hyperparameters (Keras Tuner)...")
    tuner_cls_lstm.search(X_train_cls_lstm, y_train_cls_lstm, epochs=10, batch_size=64,
                        validation_data=(X_test_cls_lstm, y_test_cls_lstm), verbose=0,
                        callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=3)])

    best_hp_cls_lstm_kt = tuner_cls_lstm.get_best_hyperparameters(num_trials=1)[0]
    logger.info(f"LSTM Classifier Keras Tuner Best Hyperparameters: {best_hp_cls_lstm_kt.values}")
    best_model_cls_lstm_kt = tuner_cls_lstm.get_best_models(num_models=1)[0]
    kt_loss, kt_acc = best_model_cls_lstm_kt.evaluate(X_test_cls_lstm, y_test_cls_lstm, verbose=0)
    logger.info(f"LSTM Classifier Keras Tuner Best Model Eval - Loss: {kt_loss:.4f}, Accuracy: {kt_acc:.4f}")

    # 2. Optuna (fine-tuning) - Define objective function inside the try block
    def objective_lstm_cls(trial):
        # Search space around Keras Tuner best params
        units = trial.suggest_int('units',
                                  max(16, best_hp_cls_lstm_kt.get('units') - 16),
                                  best_hp_cls_lstm_kt.get('units') + 16)
        dropout_rate = trial.suggest_float('dropout_rate',
                                            max(0.0, best_hp_cls_lstm_kt.get('dropout_rate') - 0.1),
                                            min(0.7, best_hp_cls_lstm_kt.get('dropout_rate') + 0.1), step=0.05)
        learning_rate = trial.suggest_float('learning_rate',
                                              best_hp_cls_lstm_kt.get('learning_rate') * 0.1,
                                              best_hp_cls_lstm_kt.get('learning_rate') * 10, log=True)

        model = Sequential()
        model.add(LSTM(units=units, return_sequences=False, input_shape=(X_train_cls_lstm.shape[1], X_train_cls_lstm.shape[2])))
        model.add(Dropout(dropout_rate))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(optimizer=Adam(learning_rate=learning_rate),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        # Using simple train/validation split
        history = model.fit(X_train_cls_lstm, y_train_cls_lstm, epochs=5, batch_size=64,
                            validation_split=0.2,
                            verbose=0, callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=2)])

        # Return the best validation accuracy achieved during training for this trial
        val_accuracy = max(history.history.get('val_accuracy', [0]))
        return val_accuracy


    logger.info("Running Optuna tuning for LSTM Classifier...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study_lstm_cls = optuna.create_study(direction='maximize')
    study_lstm_cls.optimize(objective_lstm_cls, n_trials=20)
    optuna_best_params_cls_lstm = study_lstm_cls.best_params
    logger.info(f"LSTM Classifier Optuna Best Hyperparameters: {optuna_best_params_cls_lstm}")
    logger.info(f"LSTM Classifier Optuna Best Score (CV Accuracy): {study_lstm_cls.best_value:.4f}")

    # Evaluate Final Model (using Optuna best params)
    final_model_cls_lstm = Sequential()
    final_model_cls_lstm.add(LSTM(units=optuna_best_params_cls_lstm['units'], return_sequences=False, input_shape=(X_train_cls_lstm.shape[1], X_train_cls_lstm.shape[2])))
    final_model_cls_lstm.add(Dropout(optuna_best_params_cls_lstm['dropout_rate']))
    final_model_cls_lstm.add(Dense(num_classes, activation='softmax'))
    final_model_cls_lstm.compile(optimizer=Adam(learning_rate=optuna_best_params_cls_lstm['learning_rate']),
                                loss='sparse_categorical_crossentropy',
                                metrics=['accuracy'])

    logger.info("Training final LSTM Classifier model with Optuna parameters...")
    final_model_cls_lstm.fit(X_train_cls_lstm, y_train_cls_lstm, epochs=20, batch_size=64,
                            validation_data=(X_test_cls_lstm, y_test_cls_lstm), verbose=0,
                            callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=5)])


    loss_cls_lstm, acc_cls_lstm = final_model_cls_lstm.evaluate(X_test_cls_lstm, y_test_cls_lstm, verbose=0)
    predictions_cls_lstm = final_model_cls_lstm.predict(X_test_cls_lstm)
    y_pred_cls_lstm = np.argmax(predictions_cls_lstm, axis=1)
    f1_cls_lstm = f1_score(y_test_cls_lstm, y_pred_cls_lstm, average='weighted')

    # Store results
    results['LSTM_Classifier'] = {
        'Best Params (Optuna)': optuna_best_params_cls_lstm,
        'Test Loss': loss_cls_lstm,
        'Test Accuracy': acc_cls_lstm,
        'Test F1 (weighted)': f1_cls_lstm
    }
    logger.info(f"LSTM Classifier Metrics - Test Loss: {loss_cls_lstm:.4f}, Accuracy: {acc_cls_lstm:.4f}, F1: {f1_cls_lstm:.4f}")

except Exception as e:
    logger.error(f"Error during LSTM Classifier execution: {e}", exc_info=True)
    results['LSTM_Classifier'] = {'Error': str(e)}
end_time = time.time()
logger.info(f"LSTM Classifier (Task 2A) took {end_time - start_time:.2f} seconds.")



# --- Comparison Summary ---
logger.info("--- Comparison Summary ---")

print("\n" + "="*60)
print("           Model Comparison Summary")
print("="*60 + "\n")

# --- Random Forest Comparison ---
print("-" * 20 + " Random Forest Comparison " + "-" * 20)
rf_reg_res = results.get('RF_Regressor', {})
rf_cls_res = results.get('RF_Classifier', {})

if 'Error' in rf_reg_res:
    print("Random Forest Regressor (Task 4): Execution failed.")
    print(f"  Error: {rf_reg_res['Error']}")
else:
    print("Random Forest Regressor (Task 4 - Predicts Continuous Mood):")
    print(f"  Best Params (GridSearch): {rf_reg_res.get('Best Params', 'N/A')}")
    print(f"  Mean Absolute Error (MAE): {rf_reg_res.get('MAE', 'N/A'):.4f}")
    print(f"  Mean Squared Error (MSE):  {rf_reg_res.get('MSE', 'N/A'):.4f}")
    print(f"  R-squared (R2):          {rf_reg_res.get('R2', 'N/A'):.4f}")

print("-" * 5) # Separator

if 'Error' in rf_cls_res:
    print("Random Forest Classifier (Task 2A): Execution failed.")
    print(f"  Error: {rf_cls_res['Error']}")
else:
    print("Random Forest Classifier (Task 2A - Predicts Mood Category):")
    print(f"  Best Params (Optuna): {rf_cls_res.get('Best Params (Optuna)', 'N/A')}")
    print(f"  Accuracy:             {rf_cls_res.get('Accuracy', 'N/A'):.4f}")
    print(f"  F1 Score (weighted):  {rf_cls_res.get('F1 (weighted)', 'N/A'):.4f}")
    print(f"  MSE (on classes):     {rf_cls_res.get('MSE', 'N/A'):.4f}") # MSE/R2 less meaningful here
    print(f"  R2 (on classes):      {rf_cls_res.get('R2', 'N/A'):.4f}")

print("\nDiscussion (Random Forest):")
if 'Error' not in rf_reg_res and 'Error' not in rf_cls_res:
    print("- RF Regressor aims for precise mood values (lower MAE/MSE is better).")
    print("- RF Classifier aims for correct mood categories (higher Accuracy/F1 is better).")
    print("- Feature importances might differ between the two due to the different target variables.")

    print(f"- The classifier achieved {rf_cls_res.get('Accuracy', 0)*100:.1f}% accuracy in predicting mood category.")
    print(f"- The regressor predicted mood with an average error (MAE) of {rf_reg_res.get('MAE', 0):.2f} points.")
else:
    print("- Comparison incomplete due to execution errors in one or both models.")

print("\n" + "="*60 + "\n")


# --- LSTM Comparison ---
print("-" * 25 + " LSTM Comparison " + "-" * 25)
lstm_reg_res = results.get('LSTM_Regressor', {})
lstm_cls_res = results.get('LSTM_Classifier', {})

if 'Error' in lstm_reg_res:
    print("LSTM Regressor (Task 4): Execution failed.")
    print(f"  Error: {lstm_reg_res['Error']}")
else:
    print("LSTM Regressor (Task 4 - Predicts Continuous Mood):")
    print(f"  Best Params (Keras Tuner): {lstm_reg_res.get('Best Params', 'N/A')}")
    print(f"  Test MAE:                {lstm_reg_res.get('Test MAE', 'N/A'):.4f}")
    print(f"  Test MSE:                {lstm_reg_res.get('Test MSE', 'N/A'):.4f}")
    print(f"  Test R2:                 {lstm_reg_res.get('Test R2', 'N/A'):.4f}")
    print(f"  Test Loss (MAE):         {lstm_reg_res.get('Test Loss (MAE)', 'N/A'):.4f}")

print("-" * 5)

if 'Error' in lstm_cls_res:
    print("LSTM Classifier (Task 2A): Execution failed.")
    print(f"  Error: {lstm_cls_res['Error']}")
else:
    print("LSTM Classifier (Task 2A - Predicts Mood Category):")
    print(f"  Best Params (Optuna):    {lstm_cls_res.get('Best Params (Optuna)', 'N/A')}")
    print(f"  Test Accuracy:           {lstm_cls_res.get('Test Accuracy', 'N/A'):.4f}")
    print(f"  Test F1 Score (weighted):{lstm_cls_res.get('Test F1 (weighted)', 'N/A'):.4f}")
    print(f"  Test Loss (CrossEntropy):{lstm_cls_res.get('Test Loss', 'N/A'):.4f}")

print("\nDiscussion (LSTM):")
if 'Error' not in lstm_reg_res and 'Error' not in lstm_cls_res:
    print("- LSTM Regressor uses temporal sequences to predict precise mood values (lower MAE/MSE is better).")
    print("- LSTM Classifier uses temporal sequences to predict mood categories (higher Accuracy/F1 is better).")
    print("- The final layer activation ('linear' vs 'softmax') and loss function ('mae' vs 'sparse_categorical_crossentropy') reflect the different tasks.")

    print(f"- The LSTM classifier achieved {lstm_cls_res.get('Test Accuracy', 0)*100:.1f}% accuracy on the test set.")
    print(f"- The LSTM regressor predicted mood with an average error (MAE) of {lstm_reg_res.get('Test MAE', 0):.2f} points on the test set.")
else:
    print("- Comparison incomplete due to execution errors in one or both models.")


print("\n" + "="*60)
print("           Overall Model Type Comparison")
print("="*60 + "\n")

# Compare RF vs LSTM within each task
print("-" * 15 + " Regression Task (RF Reg vs LSTM Reg) " + "-" * 15)
if 'Error' not in rf_reg_res and 'Error' not in lstm_reg_res:
    rf_mae = rf_reg_res.get('MAE', float('inf'))
    lstm_mae = lstm_reg_res.get('Test MAE', float('inf'))
    rf_r2 = rf_reg_res.get('R2', float('-inf'))
    lstm_r2 = lstm_reg_res.get('Test R2', float('-inf'))
    print(f"  RF Regressor MAE:   {rf_mae:.4f} | R2: {rf_r2:.4f}")
    print(f"  LSTM Regressor MAE:  {lstm_mae:.4f} | R2: {lstm_r2:.4f}")
    if lstm_mae < rf_mae:
        print("  Conclusion: LSTM Regressor performed better (lower MAE).")
        if rf_mae != float('inf'): print(f"    MAE Difference: {rf_mae - lstm_mae:.4f}")
    elif rf_mae < lstm_mae:
        print("  Conclusion: Random Forest Regressor performed better (lower MAE).")
        if lstm_mae != float('inf'): print(f"    MAE Difference: {lstm_mae - rf_mae:.4f}")
    else:
        print("  Conclusion: Both models performed similarly based on MAE.")
else:
    print("  Comparison not possible due to errors.")


print("\n" + "-" * 15 + " Classification Task (RF Cls vs LSTM Cls) " + "-" * 14)
if 'Error' not in rf_cls_res and 'Error' not in lstm_cls_res:
    rf_acc = rf_cls_res.get('Accuracy', 0)
    lstm_acc = lstm_cls_res.get('Test Accuracy', 0)
    rf_f1 = rf_cls_res.get('F1 (weighted)', 0)
    lstm_f1 = lstm_cls_res.get('Test F1 (weighted)', 0)
    print(f"  RF Classifier Accuracy: {rf_acc:.4f} | F1: {rf_f1:.4f}")
    print(f"  LSTM Classifier Accuracy: {lstm_acc:.4f} | F1: {lstm_f1:.4f}")
    if lstm_acc > rf_acc:
        print("  Conclusion: LSTM Classifier performed better (higher Accuracy).")
        print(f"    Accuracy Difference: {lstm_acc - rf_acc:.4f}")
    elif rf_acc > lstm_acc:
        print("  Conclusion: Random Forest Classifier performed better (higher Accuracy).")
        print(f"    Accuracy Difference: {rf_acc - lstm_acc:.4f}")
    else:
        print("  Conclusion: Both models performed similarly based on Accuracy.")
else:
    print("  Comparison not possible due to errors.")


print("\n" + "="*60)
print("               End of Summary")
print("="*60)