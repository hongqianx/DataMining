from common.helpers import logger
import numpy as np
from sklearn.preprocessing import LabelEncoder

# --- Consntants ---
MAX_PRICE_NIGHT = 150000

# Generate additional features that may be useful for the model
def feature_engineering(data):
    logger.debug("Running feature engineering")

    # Feature for total number of adults and children
    data["total_people"] = data["srch_adults_count"] + data["srch_children_count"]

    # History differences
    data["history_starrating_diff"] = data["visitor_hist_starrating"] - data["prop_starrating"]
    data["history_adr_diff"] = data["visitor_hist_adr_usd"] - data["price_usd"]
    data["price_history_difference"] = data["prop_log_historical_price"] - np.log1p(data["price_usd"])

    # Total price per night per room
    data['price_1room_1night'] = (data['price_usd'] / data['srch_room_count']) / data['srch_length_of_stay']
    data.drop(columns=['price_usd'], inplace=True)

    # Filter out high prices
    data = data[data['price_1room_1night'] < MAX_PRICE_NIGHT].copy()

    # log transform
    data['price_1room_1night_log'] = np.log1p(data['price_1room_1night'])
    data.drop(columns=['price_1room_1night'], inplace=True)

    # log transform price, min price is 0, so we use log(x+1)
    data['visitor_hist_adr_usd_log'] = np.log1p(data['visitor_hist_adr_usd'])
    data.drop(columns=['visitor_hist_adr_usd'], inplace=True)

    # Transformations of competitor rates
    data["avg_comp_rate"] = data[["comp1_rate", "comp2_rate", "comp3_rate", "comp4_rate", "comp5_rate", "comp6_rate", "comp7_rate", "comp8_rate"]].sum(axis=1)
    data["avg_comp_inv"] = data[["comp1_inv", "comp2_inv", "comp3_inv", "comp4_inv", "comp5_inv", "comp6_inv", "comp7_inv", "comp8_inv"]].sum(axis=1)
    data["avg_comp_rate_percent_diff"] = data[["comp1_rate_percent_diff", "comp2_rate_percent_diff", "comp3_rate_percent_diff", "comp4_rate_percent_diff", "comp5_rate_percent_diff", "comp6_rate_percent_diff", "comp7_rate_percent_diff", "comp8_rate_percent_diff"]].dropna().median(axis=1)

    # Locational features
    data["domestic_travel_bool"] = data["prop_country_id"] == data["visitor_location_country_id"]

    # Transform columns to unique category labels
    cols_to_encode = ['site_id', 'visitor_location_country_id',
                    'prop_country_id', 'prop_id', 'srch_destination_id']
    le = LabelEncoder()
    for col in cols_to_encode:
        data[col] = le.fit_transform(data[col].astype(str))
    
    # drop original competitor columns
    for x in range(1, 9):
        data.drop(columns=[f"comp{x}_rate", f"comp{x}_inv", f"comp{x}_rate_percent_diff"], inplace=True, errors='ignore')

    data.drop(columns=['srch_destination_id'], inplace=True, errors='ignore')
    
    logger.debug("Feature engineering completed")
    return data