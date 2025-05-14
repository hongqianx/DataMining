import numpy as np
import pandas as pd
import datetime as dt
from lightgbm import LGBMRanker
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from common.helpers import logger, has_nvidia_gpu, display_feature_importances
from common.feature_engineering import feature_engineering
from common.imputation import get_imputation_values, apply_imputation

HAS_GPU = False # has_nvidia_gpu()
TESTSPLIT_RATIO = 10 
TRAIN_WITHOUT_EVALUATION = False 
TRAIN_DATA_PERCENTAGE = 1 
TEST_DATA_PERCENTAGE = 1 

training_data_path = r"../input/training_set_VU_DM.csv"
test_data_path = r"../input/test_set_VU_DM.csv"
df = pd.read_csv(training_data_path).sample(frac=TRAIN_DATA_PERCENTAGE, random_state=42)
df_test = pd.read_csv(test_data_path).sample(frac=TEST_DATA_PERCENTAGE, random_state=42)

print(df.head())

logger.info("Starting preprocessing pipeline")

impute_values = get_imputation_values(df)

df = apply_imputation(df, impute_values)
df = feature_engineering(df)

# conditions = [
#     df['booking_bool'] == 1,
#     df['click_bool'] == 1
# ]

# choices = [5, 1]

# # df['book_feature'] = np.select(conditions, choices, default=0)
# df.drop(columns=['booking_bool', 'click_bool'], inplace=True, errors='ignore')

# NOTE ASSIGNMENT PROVIDED TEST SET CONTAINS NO CLICK_BOOL THUS USELESS FOR TESTING 
df_test = apply_imputation(df_test, impute_values)
df_test = feature_engineering(df_test)

logger.info("Preprocessing pipeline finished.")

target_value = "booking_bool"
exclude_values = [target_value] + ["position", "gross_bookings_usd", "click_bool"]
target_col = df[target_value]
df.sort_values(by='srch_id', inplace=True)
df.set_index("srch_id", inplace=True)

X = df.drop(columns=exclude_values)
y = target_col

# NOTE SINCE ASSIGNMENT PROVIDED TEST SET CONTAINS NO CLICK_BOOL, WE USE THE TRAINING SET FOR TESTING
if TRAIN_WITHOUT_EVALUATION:
    x_train = X
    x_test_kaggle = df_test # Renaming to avoid confusion with internal test set
    y_train = y
    y_test = None
else:
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=TESTSPLIT_RATIO/100, random_state=42)
    x_test_kaggle = df_test # This is the externally provided test set for submission

get_group_size = lambda df: df.reset_index().groupby("srch_id")['srch_id'].count()
train_groups = get_group_size(x_train)
test_groups = get_group_size(x_test)

lgbm_ranker_params = {
    'objective': 'lambdarank',
    'learning_rate': 0.05,
    'num_leaves': 100,
    'n_estimators': 300,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'n_jobs': -1
}

if HAS_GPU:
    lgbm_ranker_params['device_type'] = 'gpu'

lgbm_ranker_model = LGBMRanker(**lgbm_ranker_params)

logger.info(f"Preparing data for LGBMRanker training. x_train shape: {x_train.shape}")

# train_groups = x_train.groupby('srch_id').size().to_list()
# x_train_for_fit = x_train.drop(columns=['srch_id'])

logger.info(f"Start training LGBMRanker at {dt.datetime.now()}")
# Eval set = list of all srch id.
# Eval group = list of occurences of srch id in our results.
lgbm_ranker_model.fit(
    x_train,
    y_train,
    group=train_groups,
    eval_set=[(x_test, y_test)],
    eval_group=[test_groups],
    eval_metric=['map']
)
logger.info(f"LGBMRanker training finished at {dt.datetime.now()}")

feature_column_names = x_train.columns.tolist()
display_feature_importances(lgbm_ranker_model, feature_column_names, "LGBMRanker")

if not TRAIN_WITHOUT_EVALUATION and y_test is not None:
    logger.info(f"Predicting on internal test set. x_test shape: {x_test.shape}")
    if 'srch_id' in x_test.columns:
        x_test_for_predict = x_test.drop(columns=['srch_id'])
    else:
        x_test_for_predict = x_test # Should not happen if x_test comes from X
    
    predictions_on_test_set = lgbm_ranker_model.predict(x_test_for_predict)

    model_rmse = np.sqrt(np.mean((predictions_on_test_set - y_test)**2))
    print(f"LGBMRanker model RMSE on internal test set: {model_rmse}")
    
    y_pred_class = (predictions_on_test_set >= 0.5).astype(int)

    mae = mean_absolute_error(y_test, predictions_on_test_set)
    mse = mean_squared_error(y_test, predictions_on_test_set)
    r2 = r2_score(y_test, predictions_on_test_set)
    
    try:
        acc = accuracy_score(y_test, y_pred_class)
        logger.info(f"Test Accuracy: {acc*100:.2f}%")
    except ValueError as e:
        logger.warning(f"Could not calculate accuracy. Error: {e}. This might happen if y_test and y_pred_class have different types or shapes unexpectedly.")


    logger.info(f"Test MAE: {mae:.4f}")
    logger.info(f"Test MSE: {mse:.4f}")
    logger.info(f"Test R²: {r2:.4f}")


logger.info("Preparing Kaggle test set for final prediction")
print(X.columns)
print(x_test_kaggle.columns)
df_test_submission_features = x_test_kaggle[X.columns.intersection(x_test_kaggle.columns)]

if 'srch_id' not in df_test_submission_features.columns:
    logger.error("'srch_id' not found in Kaggle test set features. This is required for submission grouping but not for model prediction if already dropped.")
    # If srch_id was critical for feature engineering that relied on its presence AND it's missing now, that's an issue.
    # For prediction features, it should be dropped.
    df_test_submission_features_for_predict = df_test_submission_features
else:
    df_test_submission_features_for_predict = df_test_submission_features.drop(columns=['srch_id'])


logger.info(f"Generating predictions on Kaggle test set. Features shape: {df_test_submission_features_for_predict.shape}")
kaggle_predictions = lgbm_ranker_model.predict(df_test_submission_features_for_predict)
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

create_submission_file(x_test_kaggle, kaggle_predictions, "my_kaggle_submission.csv")

logger.info("Script finished.")