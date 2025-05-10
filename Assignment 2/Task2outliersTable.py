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

# Compose table with characteristics about columns
def validate_columns(df: pd.DataFrame, rules: dict = None):
    rules = rules or {}
    results = []

    for col in df.columns:
        series = df[col]
        result = {"column": col}
        result["has_nan"] = series.isna().any()

        col_rules = rules.get(col, {})

        if pd.api.types.is_numeric_dtype(series) and ("min" in col_rules or "max" in col_rules):
            outliers = pd.Series(False, index=series.index)
            if "min" in col_rules:
                outliers |= series < col_rules["min"]
            if "max" in col_rules:
                outliers |= series > col_rules["max"]
            result["has_outliers"] = outliers.any()
            result["min"] = col_rules.get("min")
            result["max"] = col_rules.get("max")
        else:
            result["has_outliers"] = None
            result["min"] = None
            result["max"] = None

        results.append(result)

    return pd.DataFrame(results)

summary = validate_columns(
    df,
    rules={
        "prop_starrating": {"min": 0, "max": 5},
        "prop_review_score": {"min": 0, "max": 5},
        "prop_brand_bool": {"min": 0, "max": 1},
        "promotion_flag": {"min": 0, "max": 1},
        "comp1_rate": {"min": -1, "max": 1},
        "srch_adults_count": {"min": 1},
        "srch_room_count": {"min": 1}
    }
)

print(summary)