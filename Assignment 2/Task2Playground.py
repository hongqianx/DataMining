# File to test characteristics of the dataset
import logging
import pandas as pd
import matplotlib.pyplot as plt
import sys

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
input_data = r"../input/training_set_VU_DM.csv"
df = pd.read_csv(input_data)

# Test various characteristics of the dataset

## Are there star rating outliers? (outside 1 to 5, 0 means no stars)
def test_star_rating_outliers(data):
    # Check for outliers in the star rating
    outliers = data[(data["prop_starrating"] < 0) | (data["prop_starrating"] > 5)]
    if not outliers.empty:
        logger.warning(f"Star rating outliers found: {len(outliers)}")
    else:
        logger.info("No star rating outliers found.")
test_star_rating_outliers(df)

## Are there outliers in the review scores?
def test_review_score_outliers(data):
    logger.info("Max review score: %s", data["prop_review_score"].max())
    outliers = data[(data["prop_review_score"] < 0) | (data["prop_review_score"] > 5)]
    if not outliers.empty:
        logger.warning(f"Review score outliers found: {len(outliers)}")
    else:
        logger.info("No review score outliers found.")
    # Check if reviews contain NAN
    if data["prop_review_score"].isnull().any():
        logger.warning("NAN values found in review scores.")
    else:
        logger.info("No NAN values found in review scores.")
test_review_score_outliers(df)

## Does history starrating have default zero occurences?
def test_history_starrating(data):
    col = "visitor_hist_starrating"

    # Check for NaN values
    nan_count = data[col].isna().sum()
    logger.info(f"Total NaN count: {nan_count}")

    # Check for zero values (real zeros) => Result: there were 0 real zeros, so we can replace nan with 0.
    zero_count = (data[col] == 0).sum()
    logger.info(f"Total zero occurrences (real 0s, not NaN): {zero_count}")
test_history_starrating(df)

## Does history adr usd have default zero occurences?
def test_history_starrating(data):
    col = "visitor_hist_starrating"

    # Check for NaN values
    nan_count = data[col].isna().sum()
    logger.info(f"Total NaN count: {nan_count}")

    # Check for zero values (real zeros) => Result: there were 0 real zeros, so we can replace nan with 0.
    zero_count = (data[col] == 0).sum()
    logger.info(f"Total zero occurrences (real 0s, not NaN): {zero_count}")
test_history_starrating(df)
