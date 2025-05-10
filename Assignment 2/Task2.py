import logging
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# Generate additional features that may be useful for the model
def feature_engineering(data):
    # Feature for total number of adults and children
    data["total_people"] = data["srch_adults_count"] + data["srch_children_count"]
    
    # Total price per night
    data["price_per_night"] = data["price_usd"] / data["total_nights"]

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


df = feature_engineering(df)
df = handle_outliers(df, df.columns)

