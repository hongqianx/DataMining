from common.helpers import logger

def get_imputation_values(train_data):
    logger.debug("Calculating imputation values")
    
    # Impute missing values with median or specific values
    impute_values = {
        "visitor_hist_starrating": train_data["visitor_hist_starrating"].median(),
        "visitor_hist_adr_usd": train_data["visitor_hist_adr_usd"].median(),
        "prop_review_score": 0,
        "prop_location_score2": 0,
        "srch_query_affinity_score": train_data["srch_query_affinity_score"].min(),
        "orig_destination_distance": train_data["orig_destination_distance"].median(),
        "price_usd_cap": 20000,
        "price_usd_median": train_data["price_usd"].median(),
    }

    for x in range(1, 9):
        impute_values[f"comp{x}_rate"] = train_data[f"comp{x}_rate"].median()
        impute_values[f"comp{x}_inv"] = train_data[f"comp{x}_inv"].median()
        impute_values[f"comp{x}_rate_percent_diff"] = train_data[f"comp{x}_rate_percent_diff"].median()
    
    return impute_values

def apply_imputation(data, impute_values, gpu=False):
    logger.debug("Applying imputation")
    df = data.copy()

    df["visitor_hist_starrating"] = df["visitor_hist_starrating"].fillna(impute_values["visitor_hist_starrating"])
    df["visitor_hist_adr_usd"] = df["visitor_hist_adr_usd"].fillna(impute_values["visitor_hist_adr_usd"])
    df["prop_review_score"] = df["prop_review_score"].fillna(impute_values["prop_review_score"])
    df["prop_location_score2"] = df["prop_location_score2"].fillna(impute_values["prop_location_score2"])
    df["srch_query_affinity_score"] = df["srch_query_affinity_score"].fillna(impute_values["srch_query_affinity_score"])
    df["orig_destination_distance"] = df["orig_destination_distance"].fillna(impute_values["orig_destination_distance"])

    # GPU dataframes cannot handle loc function. 
    if (gpu):
        df["price_usd"] = df["price_usd"].clip(upper=impute_values["price_usd_cap"])
    else:
        df.loc[df["price_usd"] > impute_values["price_usd_cap"], "price_usd"] = impute_values["price_usd_median"]

    df = df.drop(columns=["date_time"])

    logger.debug("Imputation finished.")
    return df