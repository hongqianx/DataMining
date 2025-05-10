import logging
import sys
import numpy as np
import optuna
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

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

# Generate additional features that may be useful for the model
def feature_engineering(data):
    # Feature for total number of adults and children
    data["total_people"] = data["srch_adults_count"] + data["srch_children_count"]
    
    # Total price per night
    data["price_per_night"] = data["price_usd"] / data["srch_length_of_stay"]

    # History differences
    data["history_starrating_diff"] = data["visitor_hist_starrating"] - data["prop_starrating"]
    data["history_adr_diff"] = data["visitor_hist_adr_usd"] - data["price_usd"]

    # Transformations of competitor rates
    data["avg_comp_rate"] = data[["comp1_rate", "comp2_rate", "comp3_rate", "comp4_rate", "comp5_rate", "comp6_rate", "comp7_rate", "comp8_rate"]].mean(axis=1)
    data["avg_comp_inv"] = data[["comp1_inv", "comp2_inv", "comp3_inv", "comp4_inv", "comp5_inv", "comp6_inv", "comp7_inv", "comp8_inv"]].mean(axis=1)
    data["avg_comp_rate_percent_diff"] = data[["comp1_rate_percent_diff", "comp2_rate_percent_diff", "comp3_rate_percent_diff", "comp4_rate_percent_diff", "comp5_rate_percent_diff", "comp6_rate_percent_diff", "comp7_rate_percent_diff", "comp8_rate_percent_diff"]].mean(axis=1)

    # Locational features
    data["customer_hotel_country_equal"] = data["prop_country_id"] == data["visitor_location_country_id"]

    return data

# For outliers in the data, handle them individually
def handle_outliers(data, columns, upper_percent=0.99):
    df_trim = data.copy()
    PRICE_OUTLIER = 20000

  # Handle unknown values of all columns per-case
    df_trim["visitor_hist_starrating"] = df_trim["visitor_hist_starrating"].fillna((df_trim["visitor_hist_starrating"].median()))
    df_trim["visitor_hist_adr_usd"] = df_trim["visitor_hist_adr_usd"].fillna((df_trim["visitor_hist_adr_usd"].median()))
    df_trim["prop_review_score"] = df_trim["prop_review_score"].fillna(0)
    df_trim["prop_location_score2"] = df_trim["prop_location_score2"].fillna(0)
    # Consider worst case scenario if affinity is not available
    df_trim["srch_query_affinity_score"] = df_trim["srch_query_affinity_score"].fillna(df_trim["srch_query_affinity_score"].min())
    df_trim["orig_destination_distance"] = df_trim["orig_destination_distance"].fillna(df_trim["orig_destination_distance"].median())
    for x in range(1,8):
        df_trim["comp" + str(x) + "_rate"] = df_trim["comp" + str(x) + "_rate"].fillna(df_trim["comp" + str(x) + "_rate"].median())
        df_trim["comp" + str(x) + "_inv"] = df_trim["comp" + str(x) + "_inv"].fillna(df_trim["comp" + str(x) + "_inv"].median())
        df_trim["comp" + str(x) + "_rate_percent_diff"] = df_trim["comp" + str(x) + "_rate_percent_diff"].fillna(df_trim["comp" + str(x) + "_rate_percent_diff"].median())
    df_trim.loc[df_trim["price_usd"] > PRICE_OUTLIER, "price_usd"] = df_trim["price_usd"].median()
    df_trim["gross_bookings_usd"] = df_trim["gross_bookings_usd"].fillna(df_trim["gross_bookings_usd"].median())

    return df_trim

def transform_data(data):
    data['date_time_epoch'] = pd.to_datetime(data['date_time']).apply(lambda x: x.timestamp())
    data = data.drop(columns=['date_time'])
    return data

df = transform_data(df)
df = feature_engineering(df)
df = handle_outliers(df, df.columns)

target_value = "booking_bool"
target_col = df[target_value]

# Retrieve the data into training and testing sets (splitting is already done)
x_train = df.drop(columns=[target_value])
y_train = target_col
if target_value in df_test.columns:
    x_test = df_test.drop(columns=[target_value])
    y_test = df_test[target_value]
else:
    x_test = df_test
    y_test = None
kf = KFold(n_splits=5, shuffle=True, random_state=42)

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
        X_train_cv, X_val_cv = x_train.iloc[train_index], x_train.iloc[val_index]
        y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[val_index]

        model.fit(X_train_cv.to_numpy(), y_train_cv)

        y_pred = model.predict(X_val_cv)
        rmse = np.sqrt(np.mean((y_pred - y_val_cv)**2))
        cv_rmse.append(rmse)

    # Average RMSE across folds
    avg_rmse = np.mean(cv_rmse)
    logger.info(f"Model: {model_name}, Cross-Validated RMSE: {avg_rmse}")
    return avg_rmse

def create_model(model_name, params):
    if model_name == 'xgb': return XGBRegressor(**params)
    elif model_name == 'lgbm': return LGBMRegressor(**params)
    elif model_name == 'rf': return RandomForestRegressor(**params)
    elif model_name == 'catboost': return CatBoostRegressor(**params, verbose=0)

models = ['xgb', 'lgbm', 'rf', 'catboost']


# TODO below is work in progress
# TODO use Neural network ensemble

best_models = []
for model_name in models:
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: hyperOptimization(trial, model_name), n_trials=50)

    print(f"Best hyperparameters for {model_name}: {study.best_params}")

    best_params = study.best_params
    best_model = create_model(model_name, best_params)

    best_model.fit(x_train.reshape((x_train.shape[0], 1, x_train.shape[1])), y_train, epochs=5, batch_size=32, verbose=0)
    best_models.append((model_name, best_model))

stacking_model = StackingRegressor(
    estimators=[(name, model) for name, model in best_models],
    final_estimator=RandomForestRegressor(n_estimators=50)
)

stacking_model.fit(x_train, y_train)

stacking_predictions = stacking_model.predict(x_test)

stacking_rmse = np.sqrt(np.mean((stacking_predictions - y_test)**2))
print(f"Ensemble model RMSE: {stacking_rmse}")