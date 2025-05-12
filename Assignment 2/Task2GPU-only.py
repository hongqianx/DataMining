import logging
import sys
import numpy as np
import optuna
import cudf
import cupy as cp
import datetime as dt
from cuml.ensemble import RandomForestRegressor as cuMLRandomForestRegressor
import lightgbm as lgb 
import xgboost as xgb
import catboost as cb
from cuml.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from cuml.model_selection import train_test_split
import pandas as pd

# --- GPU Check ---
try:
    HAS_GPU = cp.cuda.runtime.getDeviceCount() > 0
    if not HAS_GPU:
        raise SystemExit("No CUDA for GPU found. This script requires a GPU")
    cp.cuda.Device(0).use()
    print(f"Found {cp.cuda.runtime.getDeviceCount()} GPUs. Using device 0")
except Exception as e:
    raise SystemExit(f"GPU check or cupy import failed: {e}")

# Set up a basic logger
logger = logging.getLogger("MLLogger")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# --- Configuration ---
training_data_path = r"../input/training_set_VU_DM.csv"
test_data_path = r"../input/test_set_VU_DM.csv"
FOLD_AMOUNT = 3
TESTSPLIT_RATIO = 10
OPTUNA_TRIALS = 1
ENSEMBLE_N_ESTIMATORS = 10 # Estimators for final ensemble model
TRAIN_WITHOUT_EVALUATION = False
RANDOM_STATE = 42
DATA_PRECISION = cp.float32 # For GPU handling

# --- Data Loading (GPU) ---
logger.info("Loading data directly into DataFrames...")
try:
    df = cudf.read_csv(training_data_path)
    df_test = cudf.read_csv(test_data_path)
    logger.info("Data loaded successfully.")
except Exception as e:
     logger.critical(f"Failed to load data: {e}")

# --- Start preprocessing ---
def feature_engineering(data):
    logger.debug("Running feature engineering")
    data_fe = data.copy()

    # Feature for total number of persons
    data_fe["total_people"] = data_fe["srch_adults_count"] + data_fe["srch_children_count"]

    # Total price per night
    data_fe["srch_length_of_stay_safe"] = data_fe["srch_length_of_stay"].replace(0, 1) 
    data_fe["price_per_night"] = data_fe["price_usd"] / data_fe["srch_length_of_stay_safe"]
    data_fe = data_fe.drop(columns=["srch_length_of_stay_safe"])
    data_fe["price_per_night"] = data_fe["price_per_night"].fillna(0)

    # History differences
    data_fe["history_starrating_diff"] = data_fe["visitor_hist_starrating"] - data_fe["prop_starrating"]
    data_fe["history_adr_diff"] = data_fe["visitor_hist_adr_usd"] - data_fe["price_usd"]

    # Competitor features
    comp_cols_base = ["comp1", "comp2", "comp3", "comp4", "comp5", "comp6", "comp7", "comp8"]
    comp_rate_cols = [f"{c}_rate" for c in comp_cols_base if f"{c}_rate" in data_fe.columns]
    comp_inv_cols = [f"{c}_inv" for c in comp_cols_base if f"{c}_inv" in data_fe.columns]
    comp_rate_diff_cols = [f"{c}_rate_percent_diff" for c in comp_cols_base if f"{c}_rate_percent_diff" in data_fe.columns]

    # Transformations of competitor rates
    data_fe["avg_comp_rate"] = data_fe[comp_rate_cols].sum(axis=1).fillna(0)
    data_fe["avg_comp_inv"] = data_fe[comp_inv_cols].sum(axis=1).fillna(0)
    data_fe["avg_comp_rate_percent_diff"] = data_fe[comp_rate_diff_cols].mean(axis=1).fillna(0)
    
    # Locational features
    data_fe["customer_hotel_country_equal"] = (data_fe["prop_country_id"] == data_fe["visitor_location_country_id"])

    # Convert boolean features generated to int8
    for col in data_fe.select_dtypes(include=['bool']).columns:
         data_fe[col] = data_fe[col].astype(cp.int8)

    logger.debug("Feature engineering finished")
    return data_fe


def get_imputation_values_cudf(train_data):
    logger.debug("Calculating imputation values")
    impute_values = {}
    impute_values["visitor_hist_starrating"] = train_data["visitor_hist_starrating"].dropna().median()
    impute_values["visitor_hist_adr_usd"] = train_data["visitor_hist_adr_usd"].dropna().median()
    impute_values["prop_review_score"] = 0.0
    impute_values["prop_location_score2"] = 0.0
    impute_values["srch_query_affinity_score"] = train_data["srch_query_affinity_score"].dropna().min() 
    impute_values["orig_destination_distance"] = train_data["orig_destination_distance"].dropna().median() 
    impute_values["price_usd_cap"] = 20000.0
    impute_values["price_usd_median"] = train_data["price_usd"].dropna().median()

    for x in range(1, 9):
        for suffix in ["rate", "inv", "rate_percent_diff"]:
            impute_key = f"comp{x}_{suffix}"
            impute_values[impute_key] = 0.0
            try:
                numeric_col = cudf.to_numeric(train_data[impute_key])
                median_val = float(numeric_col.dropna().median())
                if not np.isnan(median_val):
                    impute_values[impute_key] = median_val
            except Exception as e:
                logger.warning(f"Could not calculate median for {impute_key}: {e}. Using 0")

    # Ensure all impute values are float
    for k, v in impute_values.items():
        if pd.isna(v):
            logger.warning(f"Imputation value for {k} is NaN, setting to 0.")
            impute_values[k] = 0.0
    logger.debug("Imputation values calculated.")
    return impute_values

def apply_imputation_cudf(data, impute_values):
    logger.debug("Applying imputation")
    df = data.copy()

    # Apply pre-calculated imputation values
    df["visitor_hist_starrating"] = df["visitor_hist_starrating"].fillna(impute_values["visitor_hist_starrating"])
    df["visitor_hist_adr_usd"] = df["visitor_hist_adr_usd"].fillna(impute_values["visitor_hist_adr_usd"])
    df["prop_review_score"] = df["prop_review_score"].fillna(impute_values["prop_review_score"])
    df["prop_location_score2"] = df["prop_location_score2"].fillna(impute_values["prop_location_score2"])
    df["srch_query_affinity_score"] = df["srch_query_affinity_score"].fillna(impute_values["srch_query_affinity_score"])
    df["orig_destination_distance"] = df["orig_destination_distance"].fillna(impute_values["orig_destination_distance"])

    for x in range(1, 9):
         for suffix in ["rate", "inv", "rate_percent_diff"]:
             col = f"comp{x}_{suffix}"
             impute_key = f"comp{x}_{suffix}"
             df[col] = df[col].fillna(impute_values[impute_key])

    df["price_usd"] = df["price_usd"].fillna(impute_values["price_usd_median"])
    df["price_usd"] = df["price_usd"].clip(upper=impute_values["price_usd_cap"])

    logger.debug("Imputation finished.")
    return df

def transform_data_cudf(data):
    logger.debug("Transforming datetime")
    dt_col = cudf.to_datetime(data['date_time'])
    data['date_time_epoch'] = (dt_col.astype('int64') // 10**9) # Convert to epoch
    data = data.drop(columns=['date_time'])
    logger.debug("Datetime transformation finished.")
    return data

# --- Preprocessing Pipeline Execution (GPU) ---
logger.info("Starting preprocessing pipeline (GPU)...")

impute_values = get_imputation_values_cudf(df)

df = transform_data_cudf(df)
df = apply_imputation_cudf(df, impute_values)
df = feature_engineering(df)

df_test_ids = df_test[['srch_id', 'prop_id']].copy()
df_test = transform_data_cudf(df_test)
df_test = apply_imputation_cudf(df_test, impute_values)
df_test = feature_engineering(df_test)

logger.info("Preprocessing pipeline finished.")

# --- Target and Feature Selection ---
target_value = "click_bool"
exclude_values = ["booking_bool", "position", "gross_bookings_usd", "srch_id", "prop_id"]
exclude_values_filtered = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in exclude_values]
exclude_values_for_X = [target_value] + exclude_values_filtered

feature_cols_final = [col for col in df.columns if col in df_test.columns and col not in exclude_values_for_X]

X = df[feature_cols_final].copy()
y = df[target_value].astype(DATA_PRECISION).copy()
X_kaggle_test = df_test[feature_cols_final].copy()

# Added because there were some memory issues.
del df
del df_test
logger.info("Original Dataframe deleted")
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

# --- Train/Validation split ---
if TRAIN_WITHOUT_EVALUATION:
    logger.info("Training on full dataset. No validation split.")
    x_train = X
    y_train = y
    x_val = None
    y_val = None
else:
    logger.info(f"Splitting data using {TESTSPLIT_RATIO}% for validation.")
    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=TESTSPLIT_RATIO/100, random_state=RANDOM_STATE)
    logger.info(f"Train shape: {x_train.shape}, Validation shape: {x_val.shape}")
    del X, y

kf = KFold(n_splits=FOLD_AMOUNT, shuffle=True, random_state=RANDOM_STATE)

# --- Hyperparameter Optimization (GPU) ---
def hyperOptimization(trial, model_name):
    params = {}
    num_boost_round = 100
    if model_name == 'xgb':
        num_boost_round = trial.suggest_int('xgb_n_estimators', 100, 400)
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',      
            'eta': trial.suggest_float('xgb_learning_rate', 1e-3, 0.2, log=True),
            'max_depth': trial.suggest_int('xgb_max_depth', 4, 10),
            'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.6, 1.0),
            'tree_method': 'hist', # Use hist method for GPU
            'device': 'cuda',
            'verbosity': 0,
            'random_state': RANDOM_STATE
        }
    elif model_name == 'lgbm':
        num_boost_round = trial.suggest_int('lgbm_n_estimators', 100, 400)
        params = {
            'objective': 'regression_l2', 
            'metric': 'rmse',           
            'learning_rate': trial.suggest_float('lgbm_learning_rate', 1e-3, 0.2, log=True),
            'num_leaves': trial.suggest_int('lgbm_num_leaves', 20, 80),
            'max_depth': trial.suggest_int('lgbm_max_depth', 4, 10),
            'subsample': trial.suggest_float('lgbm_subsample', 0.6, 1.0), 
            'colsample_bytree': trial.suggest_float('lgbm_colsample_bytree', 0.6, 1.0), 
            'device_type': 'GPU', #Use Cuda if can
            'verbosity': -1,
            'random_state': RANDOM_STATE,
            'n_jobs': 1 # Somewhere recommended to set to 1 for GPU, so just use that for now
        }
    elif model_name == 'rf':
         params = {
             'n_estimators': trial.suggest_int('rf_n_estimators', 50, 200),
             'max_depth': trial.suggest_int('rf_max_depth', 5, 16),
             'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 10),
             'n_streams': 1
         }
    elif model_name == 'catboost':
        iterations = trial.suggest_int('catboost_iterations', 100, 400)
        params = {
            'loss_function': 'RMSE',    
            'learning_rate': trial.suggest_float('catboost_learning_rate', 1e-3, 0.2, log=True),
            'depth': trial.suggest_int('catboost_depth', 4, 10),
            'random_state': RANDOM_STATE,
            'verbose': 0,                
            'task_type': 'GPU',          # Use GPU
            'devices': '0'               
        }
        params['iterations'] = iterations
    else:
        return float('inf')

    cv_rmse = []
    for fold_idx, (train_index, val_index) in enumerate(kf.split(x_train)):
        logger.debug(f"Training {model_name} fold {fold_idx + 1}/{FOLD_AMOUNT} for Optuna trial {trial.number}")
        train_idx_list = train_index.tolist()
        val_idx_list = val_index.tolist()

        X_train_cv = x_train.iloc[train_idx_list]
        X_val_cv = x_train.iloc[val_idx_list]
        y_train_cv = y_train.iloc[train_idx_list]
        y_val_cv = y_train.iloc[val_idx_list]

        model = None
        y_pred = None

        start_time = dt.datetime.now()
        try:
            if model_name == 'lgbm':
                # Use cupy arrays for lgb dataset
                X_train_cp = X_train_cv
                y_train_cp = y_train_cv
                X_val_cp = X_val_cv

                lgb_train_data = lgb.Dataset(X_train_cp, label=y_train_cp)
                model = lgb.train(params, lgb_train_data, num_boost_round=num_boost_round)
                y_pred = model.predict(X_val_cp) 
                del X_train_cp, y_train_cp, X_val_cp 

            elif model_name == 'xgb':
                # Use cuDF directly for DMatrix
                dtrain = xgb.DMatrix(X_train_cv, label=y_train_cv)
                dval = xgb.DMatrix(X_val_cv)
                model = xgb.train(params, dtrain, num_boost_round=num_boost_round)
                y_pred = model.predict(dval)

            elif model_name == 'catboost':
                train_pool = cb.Pool(data=X_train_cv, label=y_train_cv)
                model = cb.CatBoostRegressor(**params)
                model.fit(train_pool)
                y_pred = model.predict(X_val_cv) 

            elif model_name == 'rf':
                model = cuMLRandomForestRegressor(**params, random_state=RANDOM_STATE)
                model.fit(X_train_cv, y_train_cv)
                y_pred = model.predict(X_val_cv)

            else:
                 raise ValueError(f"Training logic missing for {model_name}")

            end_time = dt.datetime.now()
            logger.debug(f"Fold {fold_idx+1} fit duration: {end_time - start_time}")

            y_pred_cp = cp.asarray(y_pred)
            y_val_cp = y_val_cv.values if isinstance(y_val_cv, cudf.Series) else cp.asarray(y_val_cv)

            rmse = cp.sqrt(cp.mean((y_pred_cp - y_val_cp)**2))
            cv_rmse.append(rmse.item())

        except Exception as e:
            logger.error(f"Error during training/prediction for {model_name} trial {trial.number} fold {fold_idx+1}: {e}", exc_info=True)
            # Return a high value to Optuna if a fold fails
            return float('inf')
        finally:
            # Clean up fold data
             try: 
                del X_train_cv
                del X_val_cv
                del y_train_cv
                del y_val_cv
                del y_pred
                del y_pred_cp
                del y_val_cp
                del model
             except Exception: 
                 logger.error("Error during cleanup of fold data")
             cp.get_default_memory_pool().free_all_blocks()

    avg_rmse = np.mean(cv_rmse)
    logger.info(f"Model: {model_name}, Trial: {trial.number}, Avg CV RMSE: {avg_rmse:.5f}")
    return avg_rmse

# --- Main Training Loop
models_to_optimize = ['xgb', 'lgbm', 'rf', 'catboost']
best_trained_models = []
best_params_dict = {} 

for model_name in models_to_optimize:
    logger.info(f"--- Optimizing {model_name} ---")
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=3, n_startup_trials=3)
    study = optuna.create_study(direction='minimize', pruner=pruner)
    logger.info(f"Starting Optuna study for {model_name} with {OPTUNA_TRIALS} trials.")

    try:
        study.optimize(lambda trial: hyperOptimization(trial, model_name), n_trials=OPTUNA_TRIALS)
    except Exception as e:
        logger.error(f"Optimization failed for {model_name}: {e}", exc_info=True)
        continue # Skip to next model if optimization fails

    logger.info(f"Best hyperparameters for {model_name}: {study.best_params}")
    logger.info(f"Best CV RMSE for {model_name}: {study.best_value:.5f}")
    best_params_dict[model_name] = study.best_params

    logger.info(f"Training final {model_name} with best hyperparameters on full training data")
    start_time = dt.datetime.now()
    final_model = None
    try:
        prefix = model_name + "_"
        current_best_params = {
            key.replace(prefix, ''): value
            for key, value in study.best_params.items()
        }

        if model_name == 'xgb':
            params = {'objective': 'reg:squarederror', 'tree_method': 'hist', 'device': 'cuda', 'verbosity': 0, 'random_state': RANDOM_STATE}
            params.update(current_best_params) # Add tuned params
            num_boost_round = params.pop('n_estimators', 100) # Get n_estimators, remove from params dict for xgb.train
            dtrain = xgb.DMatrix(x_train, label=y_train)
            final_model = xgb.train(params, dtrain, num_boost_round=num_boost_round)
        elif model_name == 'lgbm':
            params = {'objective': 'regression_l2', 'metric': 'rmse', 'device_type': 'gpu', 'verbosity': -1, 'random_state': RANDOM_STATE, 'n_jobs': 1}
            params.update(current_best_params)
            num_boost_round = params.pop('n_estimators', 100)

            x_train_cp = x_train.to_cupy()
            y_train_cp = y_train.to_cupy()
            lgb_train_data = lgb.Dataset(data=x_train_cp, label=y_train_cp)

            # TODO Cuda not working for lgbm
            final_model = lgb.train(params, lgb_train_data, num_boost_round=num_boost_round)
        elif model_name == 'catboost':
            params = {'loss_function': 'RMSE', 'verbose': 0, 'random_state': RANDOM_STATE, 'task_type': 'GPU', 'devices': '0'}
            params.update(current_best_params)
            iterations = params.pop('iterations', 100) 
            train_pool = cb.Pool(data=x_train, label=y_train)
            model_instance = cb.CatBoostRegressor(**params, iterations=iterations) 
            model_instance.fit(train_pool)
            final_model = model_instance
        elif model_name == 'rf':
             params = {'n_streams': 1, 'random_state': RANDOM_STATE} 
             params.update(current_best_params)
             model_instance = cuMLRandomForestRegressor(**params)
             model_instance.fit(x_train, y_train)
             final_model = model_instance

        end_time = dt.datetime.now()
        logger.info(f"Final {model_name} trained in {end_time - start_time}.")
        if final_model:
            best_trained_models.append((model_name, final_model))
        else:
            logger.error(f"Final model training failed for {model_name}, model object is None.")

    except Exception as e:
        logger.error(f"Error during final model training for {model_name}: {e}", exc_info=True)

    # Clear memory
    cp.get_default_memory_pool().free_all_blocks()

# --- Stacking Ensemble (Manual GPU) ---
if not best_trained_models:
    raise SystemExit("No base models were trained successfully. Cannot proceed with stacking")

logger.info(f"--- Building ensemble with {len(best_trained_models)} base models ---")

meta_features_train_list = []

for name, model in best_trained_models:
    logger.debug(f"Predicting with base model: {name} on training data")
    y_pred = None
    try:
        if name == 'lgbm':
            y_pred = model.predict(x_train.to_cupy())
        elif name == 'xgb':
            dtrain_pred = xgb.DMatrix(x_train)
            y_pred = model.predict(dtrain_pred)
        elif name == 'catboost':
            y_pred = model.predict(x_train)
        elif name == 'rf':
            y_pred = model.predict(x_train)

        preds_cp = cp.asarray(y_pred)
        meta_features_train_list.append(preds_cp)
        del y_pred, preds_cp
        cp.get_default_memory_pool().free_all_blocks()
    except Exception as e:
        logger.error(f"Failed to get predictions from base model {name} on train data: {e}", exc_info=True)
        raise RuntimeError(f"Failed prediction for stacking from model {name}") from e

meta_features_train = cp.column_stack(meta_features_train_list)
del meta_features_train_list
logger.info(f"Training meta-features shape: {meta_features_train.shape}")

# Define and train the final ensemble
ensemble_model = cuMLRandomForestRegressor(
    n_estimators=ENSEMBLE_N_ESTIMATORS,
    max_depth=10,
    min_samples_split=5,
    random_state=RANDOM_STATE,
    n_streams = 2 
    )

logger.info("Training final estimator using combined models")
start_time = dt.datetime.now()
ensemble_model.fit(meta_features_train, y_train_cp)
end_time = dt.datetime.now()
logger.info(f"Final estimator trained in {end_time - start_time}.")

del x_train, y_train, y_train_cp, meta_features_train
logger.info("Deleted training data and training meta-features.")
cp.get_default_memory_pool().free_all_blocks()

# --- Evaluation (if validation set exists) ---
if not TRAIN_WITHOUT_EVALUATION and x_val is not None and y_val is not None:
    logger.info("--- Evaluating Stacking Ensemble on Validation Set ---")
    logger.info("Generating meta-features on validation data...")
    meta_features_val_list = []
    y_val_cp = y_val.values if isinstance(y_val, cudf.Series) else cp.asarray(y_val)

    for name, model in best_trained_models:
        logger.debug(f"Predicting with base model: {name} on validation data")
        y_pred = None
        try:
            if name == 'lgbm':
                y_pred = model.predict(x_val.to_cupy())
            elif name == 'xgb':
                dval_pred = xgb.DMatrix(x_val)
                y_pred = model.predict(dval_pred)
            elif name == 'catboost':
                y_pred = model.predict(x_val)
            elif name == 'rf':
                y_pred = model.predict(x_val)

            preds_cp = cp.asarray(y_pred)
            meta_features_val_list.append(preds_cp)
            del y_pred, preds_cp
            cp.get_default_memory_pool().free_all_blocks()
        except Exception as e:
             logger.error(f"Failed to get predictions from base model {name} on validation data: {e}", exc_info=True)
             raise RuntimeError(f"Failed prediction for stacking evaluation from model {name}") from e

    meta_features_val = cp.column_stack(meta_features_val_list)
    del meta_features_val_list
    logger.info(f"Validation meta-features shape: {meta_features_val.shape}")

    logger.info("Predicting with final estimator on validation meta-features...")
    stacking_predictions_val = ensemble_model.predict(meta_features_val)

    stacking_rmse = cp.sqrt(mean_squared_error(y_val_cp, stacking_predictions_val))
    stacking_mae = mean_absolute_error(y_val_cp, stacking_predictions_val)
    stacking_r2 = r2_score(y_val_cp, stacking_predictions_val)

    logger.info(f"Validation RMSE: {stacking_rmse}")
    logger.info(f"Validation MAE: {stacking_mae}")
    logger.info(f"Validation R²: {stacking_r2}")

    try:
        y_val_np = cp.asnumpy(y_val_cp)
        y_pred_class_np = cp.asnumpy(stacking_predictions_val >= 0.5).astype(int)
        acc = accuracy_score(y_val_np, y_pred_class_np)
        logger.info(f"Validation Accuracy (threshold 0.5): {acc*100:.2f}%")
        del y_val_np, y_pred_class_np
    except Exception as e:
        logger.error(f"Could not calculate accuracy: {e}")

    del x_val, y_val, y_val_cp, meta_features_val, stacking_predictions_val
    logger.info("Deleted validation data and validation meta-features.")
    cp.get_default_memory_pool().free_all_blocks()
else:
     logger.info("Skipping validation evaluation.")

# --- Kaggle Submission ---
logger.info("--- Generating Kaggle Submission ---")
logger.info("Generating meta-features on Kaggle test data (X_kaggle_test)...")
meta_features_kaggle_list = []

# Ensure X_kaggle_test has the correct columns before prediction
X_kaggle_test = X_kaggle_test[feature_cols_final]

for name, model in best_trained_models:
    logger.debug(f"Predicting with base model: {name} on Kaggle test data")
    y_pred = None
    try:
        if name == 'lgbm':
            y_pred = model.predict(X_kaggle_test.to_cupy())
        elif name == 'xgb':
            dkaggle_pred = xgb.DMatrix(X_kaggle_test)
            y_pred = model.predict(dkaggle_pred)
        elif name == 'catboost':
            y_pred = model.predict(X_kaggle_test)
        elif name == 'rf':
            y_pred = model.predict(X_kaggle_test)

        preds_cp = cp.asarray(y_pred)
        meta_features_kaggle_list.append(preds_cp)
        del y_pred, preds_cp
        cp.get_default_memory_pool().free_all_blocks()
    except Exception as e:
        logger.error(f"Failed to get predictions from base model {name} on Kaggle test data: {e}", exc_info=True)
        raise RuntimeError(f"Failed prediction for Kaggle submission from model {name}") from e

meta_features_kaggle = cp.column_stack(meta_features_kaggle_list)
del meta_features_kaggle_list

logger.info("Predicting with final ensemble on Kaggle test dataset")
kaggle_predictions_gpu = ensemble_model.predict(meta_features_kaggle)

kaggle_predictions_np = cp.asnumpy(kaggle_predictions_gpu)
logger.info("Kaggle predictions generated")

del X_kaggle_test, meta_features_kaggle, kaggle_predictions_gpu
cp.get_default_memory_pool().free_all_blocks()

# --- Submission File Creation ---
def create_submission_file(test_identifiers_cudf, predictions_numpy, output_filename="submission.csv"):
    logger.info(f"Creating submission file: {output_filename}")
    # Ensure correct format
    submission_df = test_identifiers_cudf[['srch_id', 'prop_id']].reset_index(drop=True)

    if len(submission_df) != len(predictions_numpy):
        raise ValueError(f"Length mismatch: Identifiers ({len(submission_df)}) vs Predictions ({len(predictions_numpy)})")

    pred_series = cudf.Series(predictions_numpy, name='prediction_score')
    submission_df['prediction_score'] = pred_series.astype(DATA_PRECISION)

    logger.info("Sorting submission data on GPU...")
    submission_sorted = submission_df.sort_values(
        by=['srch_id', 'prediction_score'],
        ascending=[True, False]
    )

    # Select final columns required for submission
    final_submission_gpu = submission_sorted[['srch_id', 'prop_id']]

    logger.info("Converting final submission to DataFrame")
    try:
        final_submission_pd = final_submission_gpu.to_pandas()
    except Exception as e:
        logger.error(f"Failed to convert final submission GPU to Pandas: {e}.")
        try:
            final_submission_gpu.to_csv(output_filename, index=False)
            logger.info(f"Kaggle submission file saved to: {output_filename}")
            return
        except Exception as e_cudf_save:
            logger.error(f"Direct save failed: {e_cudf_save}")

    logger.info(f"Writing submission file {output_filename}")
    final_submission_pd.to_csv(output_filename, index=False)
    logger.info(f"Kaggle submission file created successfully: {output_filename}")
    del submission_df, pred_series, submission_sorted, final_submission_gpu, final_submission_pd

create_submission_file(df_test_ids, kaggle_predictions_np, "gpu_submission.csv")
logger.info("--- Script Finished ---")