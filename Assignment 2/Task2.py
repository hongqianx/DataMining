import logging
import sys
import numpy as np
import optuna
import pandas as pd
# import cupy as cp
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, r2_score
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import subprocess

def has_nvidia_gpu():
    try:
        output = subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False

# Set up a basic logger
logger = logging.getLogger("MLLogger")
logger.setLevel(logging.DEBUG)  # Set the global logging level

# Configure logger
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Import dataset
training_data = r"../input/training_set_VU_DM.csv"
test_data = r"../input/test_set_VU_DM.csv"
df = pd.read_csv(training_data)
df_test = pd.read_csv(test_data)
HAS_GPU = has_nvidia_gpu() #HAS_GPU = cp.cuda.runtime.getDeviceCount() > 0
FOLD_AMOUNT = 3
TESTSPLIT_RATIO = 10 # Percentage of data to be used for testing
OPTUNA_TRIALS = 1 # Number of trials for hyperparameter optimization
ENSEMBLE_N_ESTIMATORS = 2 # Number of estimators for the final stacking model
TRAIN_WITHOUT_EVALUATION = True # If we should train without evaluation, gives more training data but can't output evaluation metrics

# Generate additional features that may be useful for the model
def feature_engineering(data):
    # Feature for total number of adults and children
    data["total_people"] = data["srch_adults_count"] + data["srch_children_count"]

    # log transform price, min price is 0, so we use log(x+1)
    data['visitor_hist_adr_usd_log'] = np.log1p(data['visitor_hist_adr_usd'])
    data.drop(columns=['visitor_hist_adr_usd'], inplace=True)
    
    # Total price per night per room
    data['price_1room_1night'] = (data['price_usd'] / data['srch_room_count']) / data['srch_length_of_stay']
    data.drop(columns=['price_usd'], inplace=True)
    # Filter out high prices
    data = data[data['price_1room_1night'] < 150000].copy()
    # log transform
    data['price_1room_1night_log'] = np.log1p(data['price_1room_1night'])
    data.drop(columns=['price_1room_1night'], inplace=True)


    # History differences
    data["history_starrating_diff"] = data["visitor_hist_starrating"] - data["prop_starrating"]
    data["history_adr_diff"] = data["visitor_hist_adr_usd"] - data["price_usd"]
    data["price_history_difference"] = data["prop_log_historical_price"] - np.log1p(data["price_usd"])

    # Transformations of competitor rates
    data["avg_comp_rate"] = data[["comp1_rate", "comp2_rate", "comp3_rate", "comp4_rate", "comp5_rate", "comp6_rate", "comp7_rate", "comp8_rate"]].sum(axis=1)
    data["avg_comp_inv"] = data[["comp1_inv", "comp2_inv", "comp3_inv", "comp4_inv", "comp5_inv", "comp6_inv", "comp7_inv", "comp8_inv"]].sum(axis=1)
    data["avg_comp_rate_percent_diff"] = data[["comp1_rate_percent_diff", "comp2_rate_percent_diff", "comp3_rate_percent_diff", "comp4_rate_percent_diff", "comp5_rate_percent_diff", "comp6_rate_percent_diff", "comp7_rate_percent_diff", "comp8_rate_percent_diff"]].median(axis=1)

    # Locational features
    data["domestic_travel_bool"] = data["prop_country_id"] == data["visitor_location_country_id"]

    # switch id columns to string
    cols_to_string = ['srch_id', 'site_id', 'visitor_location_country_id','prop_country_id','prop_id',\
                       'srch_destination_id']
    df[cols_to_string] = df[cols_to_string].astype('string')

    return data

# TODO Below was the old function to handle outliers, which can possibly be restored since the provided test set did not have predictory feature click_bool, thus we don't need to apply the same imputation
# For outliers in the data, handle them individually
# def handle_outliers(data, columns, upper_percent=0.99):
#     df_trim = data.copy()
#     PRICE_OUTLIER = 20000

#   # Handle unknown values of all columns per-case
#     df_trim["visitor_hist_starrating"] = df_trim["visitor_hist_starrating"].fillna((df_trim["visitor_hist_starrating"].median()))
#     df_trim["visitor_hist_adr_usd"] = df_trim["visitor_hist_adr_usd"].fillna((df_trim["visitor_hist_adr_usd"].median()))
#     df_trim["prop_review_score"] = df_trim["prop_review_score"].fillna(0)
#     df_trim["prop_location_score2"] = df_trim["prop_location_score2"].fillna(0)
#     # Consider worst case scenario if affinity is not available
#     df_trim["srch_query_affinity_score"] = df_trim["srch_query_affinity_score"].fillna(df_trim["srch_query_affinity_score"].min())
#     df_trim["orig_destination_distance"] = df_trim["orig_destination_distance"].fillna(df_trim["orig_destination_distance"].median())
#     for x in range(1,8):
#         df_trim["comp" + str(x) + "_rate"] = df_trim["comp" + str(x) + "_rate"].fillna(df_trim["comp" + str(x) + "_rate"].median())
#         df_trim["comp" + str(x) + "_inv"] = df_trim["comp" + str(x) + "_inv"].fillna(df_trim["comp" + str(x) + "_inv"].median())
#         df_trim["comp" + str(x) + "_rate_percent_diff"] = df_trim["comp" + str(x) + "_rate_percent_diff"].fillna(df_trim["comp" + str(x) + "_rate_percent_diff"].median())
#     df_trim.loc[df_trim["price_usd"] > PRICE_OUTLIER, "price_usd"] = df_trim["price_usd"].median()
#     df_trim["gross_bookings_usd"] = df_trim["gross_bookings_usd"].fillna(df_trim["gross_bookings_usd"].median())

#     return df_trim

def get_imputation_values(train_data):
    impute_values = {
        "visitor_hist_starrating": train_data["visitor_hist_starrating"].median(),
        "visitor_hist_adr_usd": train_data["visitor_hist_adr_usd"].median(),
        "prop_review_score": 0,
        "prop_location_score2": 0,
        "srch_query_affinity_score": train_data["srch_query_affinity_score"].min(),
        "orig_destination_distance": train_data["orig_destination_distance"].median(),
        "price_usd_cap": 20000,
        "price_usd_median": train_data["price_usd"].median()
    }

    for x in range(1, 9):
        impute_values[f"comp{x}_rate"] = train_data[f"comp{x}_rate"].median()
        impute_values[f"comp{x}_inv"] = train_data[f"comp{x}_inv"].median()
        impute_values[f"comp{x}_rate_percent_diff"] = train_data[f"comp{x}_rate_percent_diff"].median()
    
    return impute_values

def apply_imputation(data, impute_values):
    df = data.copy()

    df["visitor_hist_starrating"] = df["visitor_hist_starrating"].fillna(impute_values["visitor_hist_starrating"])
    df["visitor_hist_adr_usd"] = df["visitor_hist_adr_usd"].fillna(impute_values["visitor_hist_adr_usd"])
    df["prop_review_score"] = df["prop_review_score"].fillna(impute_values["prop_review_score"])
    df["prop_location_score2"] = df["prop_location_score2"].fillna(impute_values["prop_location_score2"])
    df["srch_query_affinity_score"] = df["srch_query_affinity_score"].fillna(impute_values["srch_query_affinity_score"])
    df["orig_destination_distance"] = df["orig_destination_distance"].fillna(impute_values["orig_destination_distance"])

    df.loc[df["price_usd"] > impute_values["price_usd_cap"], "price_usd"] = impute_values["price_usd_median"]

    return df

def remove_original_transformed_columns(data):
    # drop original competitor columns
    for x in range(1, 9):
        data.drop(columns=[f"comp{x}_rate", f"comp{x}_inv", f"comp{x}_rate_percent_diff"], inplace=True, errors='ignore')
    # drop columns with 10k unique value
    data.drop(columns=['srch_id','prop_id','srch_destination_id'], inplace=True, errors='ignore')
    return data

def transform_data(data):
    # currently not use search time as feature
    # data['date_time_epoch'] = pd.to_datetime(data['date_time']).apply(lambda x: x.timestamp())
    data = data.drop(columns=['date_time'])
    return data

# Determine imputation values on trainingset
impute_values = get_imputation_values(df)

# Apply the same imputation on training and test
df = transform_data(df)
df = apply_imputation(df, impute_values)
df = feature_engineering(df)
df = remove_original_transformed_columns(df)

# NOTE ASSIGNMENT PROVIDED TEST SET CONTAINS NO CLICK_BOOL THUS USELESS FOR TESTING 
df_test = transform_data(df_test)
df_test = apply_imputation(df_test, impute_values)
df_test = feature_engineering(df_test)
df_test = remove_original_transformed_columns(df_test)

target_value = "booking_bool"
exclude_values = [target_value] + ["click_bool", "position", "gross_bookings_usd"]
target_col = df[target_value]

# Retrieve the data into training and testing sets (splitting is already done)
X = df.drop(columns=exclude_values)
y = target_col

# NOTE SINCE ASSIGNMENT PROVIDED TEST SET CONTAINS NO CLICK_BOOL, WE USE THE TRAINING SET FOR TESTING
if TRAIN_WITHOUT_EVALUATION:
    x_train = X
    x_test = df_test.drop(columns=exclude_values)
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
        elif model_name == 'lgbm': return LGBMRegressor(device_type="GPU", **params)
        elif model_name == 'rf': return RandomForestRegressor(**params, n_jobs=-1)
        elif model_name == 'catboost': return CatBoostRegressor(task_type="GPU", **params, verbose=0)
    else:
        if model_name == 'xgb': return XGBRegressor(**params)
        elif model_name == 'lgbm': return LGBMRegressor(**params)
        elif model_name == 'rf': return RandomForestRegressor(**params)
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
    final_estimator=RandomForestRegressor(n_estimators=ENSEMBLE_N_ESTIMATORS)
)

logger.info(f"Training ensemble model with {len(best_models)} base models")
logger.info(f"Using the following data types for x: {x_train.dtypes} and for y: {y_train.dtypes}")
stacking_model.fit(x_train, y_train)
logger.info("Ensemble model trained successfully")

stacking_predictions = stacking_model.predict(x_test)

stacking_rmse = np.sqrt(np.mean((stacking_predictions - y_test)**2))
print(f"Ensemble model RMSE: {stacking_rmse}")

if y_test is not None:
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