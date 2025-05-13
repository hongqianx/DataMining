import numpy as np
import optuna
import pandas as pd
# import cupy as cp
import datetime as dt
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, r2_score
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from common.helpers import logger, has_nvidia_gpu
from common.feature_engineering import feature_engineering
from common.imputation import get_imputation_values, apply_imputation

# --- Configuration ---
HAS_GPU = False # has_nvidia_gpu() # False #HAS_GPU = cp.cuda.runtime.getDeviceCount() > 0
FOLD_AMOUNT = 3
TESTSPLIT_RATIO = 10 # Percentage of data to be used for testing
OPTUNA_TRIALS = 1 # Number of trials for hyperparameter optimization
ENSEMBLE_N_ESTIMATORS = 2 # Number of estimators for the final stacking model
TRAIN_WITHOUT_EVALUATION = True # If we should train without evaluation, gives more training data but can't output evaluation metrics

# --- Load the data ---
training_data_path = r"../input/training_set_VU_DM.csv"
test_data_path = r"../input/test_set_VU_DM.csv"
df = pd.read_csv(training_data_path)
df_test = pd.read_csv(test_data_path)

# --- Preprocessing Pipeline Execution ---
logger.info("Starting preprocessing pipeline")

# Determine imputation values on trainingset
impute_values = get_imputation_values(df)

# Apply the same imputation on training and test
df = apply_imputation(df, impute_values)
df = feature_engineering(df)

# NOTE ASSIGNMENT PROVIDED TEST SET CONTAINS NO CLICK_BOOL THUS USELESS FOR TESTING 
df_test = apply_imputation(df_test, impute_values)
df_test = feature_engineering(df_test)

logger.info("Preprocessing pipeline finished.")

# --- Target and Feature Selection ---
target_value = "booking_bool"
exclude_values = [target_value] + ["click_bool", "position", "gross_bookings_usd"]
target_col = df[target_value]

# Retrieve the data into training and testing sets (splitting is already done)
X = df.drop(columns=exclude_values)
y = target_col

# NOTE SINCE ASSIGNMENT PROVIDED TEST SET CONTAINS NO CLICK_BOOL, WE USE THE TRAINING SET FOR TESTING
if TRAIN_WITHOUT_EVALUATION:
    x_train = X
    x_test = df_test
    y_train = y
    y_test = None
else:
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=TESTSPLIT_RATIO/100, random_state=42)

kf = KFold(n_splits=FOLD_AMOUNT, shuffle=True, random_state=42)

# Optimize hyperparameters for ensemble models
def hyperOptimization(trial, model_name):
    model_params = {
        'xgb': {
            'learning_rate': trial.suggest_float('xgb_learning_rate', 1e-5, 1e-1, log=True),
            'max_depth': trial.suggest_int('xgb_max_depth', 3, 12),
            'n_estimators': trial.suggest_int('xgb_n_estimators', 50, 300)
        },
        'lgbm': {
            'learning_rate': trial.suggest_float('lgbm_learning_rate', 1e-5, 1e-1, log=True),
            'num_leaves': trial.suggest_int('lgbm_num_leaves', 30, 150),
            'n_estimators': trial.suggest_int('lgbm_n_estimators', 50, 300)
        },
        'rf': {
            'n_estimators': trial.suggest_int('rf_n_estimators', 50, 300),
            'max_depth': trial.suggest_int('rf_max_depth', 3, 12),
            'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 10)
        },
        'catboost': {
            'iterations': trial.suggest_int('catboost_iterations', 100, 1000),
            'learning_rate': trial.suggest_float('catboost_learning_rate', 1e-5, 1e-1, log=True),
            'depth': trial.suggest_int('catboost_depth', 3, 12)
        }
    }

    params = model_params[model_name]
    model = create_model(model_name, params)

    cv_rmse = []
    for train_index, val_index in kf.split(x_train):
        logger.info(f"Training {model_name} on fold {len(cv_rmse) + 1}")
        X_train_cv, X_val_cv = x_train.iloc[train_index], x_train.iloc[val_index]
        y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[val_index]

        logger.info(f"Start training of model {model_name}, at time {dt.datetime.now()}")
        model.fit(X_train_cv, y_train_cv)
        logger.info(f"End training of model {model_name}, at time {dt.datetime.now()}")

        y_pred = model.predict(X_val_cv)
        rmse = np.sqrt(np.mean((y_pred - y_val_cv)**2))
        cv_rmse.append(rmse)

    # Average RMSE across folds
    avg_rmse = np.mean(cv_rmse)
    logger.info(f"Model: {model_name}, Cross-Validated RMSE: {avg_rmse}")
    return avg_rmse

def create_model(model_name, params):
    if HAS_GPU:
        if model_name == 'xgb': return XGBRegressor(device = "cuda", tree_method="hist", **params)
        elif model_name == 'lgbm': return LGBMRegressor(device_type="gpu", **params)
        elif model_name == 'rf': return RandomForestRegressor(**params, n_jobs=-1)
        elif model_name == 'catboost': return CatBoostRegressor(task_type="GPU", **params, verbose=0)
    else:
        if model_name == 'xgb': return XGBRegressor(**params)
        elif model_name == 'lgbm': return LGBMRegressor(**params)
        elif model_name == 'rf': return RandomForestRegressor(**params, n_jobs=-1)
        elif model_name == 'catboost': return CatBoostRegressor(**params, verbose=0)

models = ['xgb', 'lgbm', 'rf', 'catboost']

# TODO below is work in progress
# TODO use Neural network ensemble
best_models = []
for model_name in models:
    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: hyperOptimization(trial, model_name), n_trials=OPTUNA_TRIALS)

    logger.info(f"Best hyperparameters for {model_name}: {study.best_params}")

    # Only append parameters of own model (default adds all)
    prefix = model_name + "_"
    filtered_params = {
        key.replace(prefix, ''): value
        for key, value in study.best_params.items()
        if key.startswith(prefix)
    }
    best_model = create_model(model_name, filtered_params)

    # if (model_name == 'xgb' and HAS_GPU):
    #     x_train = cp.array(cp.asarray(x_train.astype(np.float32).to_numpy()))

    logger.info(f"Training {model_name} with best hyperparameters")
    best_model.fit(x_train, y_train)
    best_models.append((model_name, best_model))
    logger.info(f"Model (best) {model_name} trained successfully")

stacking_model = StackingRegressor(
    estimators=[(name, model) for name, model in best_models],
    final_estimator=RandomForestRegressor(n_estimators=ENSEMBLE_N_ESTIMATORS, n_jobs=-1)
)

logger.info(f"Training ensemble model with {len(best_models)} base models")
logger.info(f"Using the following data types for x: {x_train.dtypes} and for y: {y_train.dtypes}")
stacking_model.fit(x_train, y_train)
logger.info("Ensemble model trained successfully")

stacking_predictions = stacking_model.predict(x_test)

# Transform results to proper submission format
logger.info("Preparing Kaggle test set for final prediction")
df_test_submission_features = df_test[X.columns]
logger.info("Generating predictions on Kaggle test set")
kaggle_predictions = stacking_model.predict(df_test_submission_features)
logger.info("Predictions for Kaggle test set generated.")

def create_submission_file(original_test_dataframe, predictions_array, output_filename="submission.csv"):
    submission = original_test_dataframe[['srch_id', 'prop_id']].copy()
    submission['prediction_score'] = predictions_array

    submission_sorted = submission.sort_values(
        by=['srch_id', 'prediction_score'],
        ascending=[True, False]
    )

    final_submission_df = submission_sorted[['srch_id', 'prop_id']]
    final_submission_df.to_csv(output_filename, index=False)
    logger.info(f"Kaggle submission file created: {output_filename}")

create_submission_file(df_test, kaggle_predictions, "my_kaggle_submission.csv")

if y_test is not None:
    stacking_rmse = np.sqrt(np.mean((stacking_predictions - y_test)**2))
    print(f"Ensemble model RMSE: {stacking_rmse}")
    y_pred = stacking_model.predict(x_test)
    y_pred_class = (y_pred >= 0.5).astype(int)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred_class)

    logger.info(f"Test MAE: {mae:.4f}")
    logger.info(f"Test MSE: {mse:.4f}")
    logger.info(f"Test R²: {r2:.4f}")
    logger.info(f"Test Accuracy: {acc*100:.2f}%")