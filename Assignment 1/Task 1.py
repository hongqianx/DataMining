import pandas as pd

# Import our data
input_data = r"../input/dataset_mood_smartphone.csv"
df = pd.read_csv(input_data)

# Clean our data
# Select the 0.1st until 99.9 quantile of our data. (581 lost)
def trim_outliers(df, lower_percent=0.01, upper_percent=0.99):
    numeric_cols = df.select_dtypes(include='number').columns
    newdf = pd.DataFrame()
    for col in numeric_cols:
        lower_bound = df[col].quantile(lower_percent)
        print(f"Lower bound for {col}: {lower_bound}")
        upper_bound = df[col].quantile(upper_percent)
        print(f"Upper bound for {col}: {upper_bound}")
        newdf[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    return newdf


df.columns.tolist()

# Time value correlation
df_with_epoch = df.assign(epoch_time=(pd.to_datetime(df['time']) - pd.Timestamp('1970-01-01')) / pd.Timedelta('1s'))
print(df_with_epoch['value'].corr(df_with_epoch['epoch_time']))

# 2. create df_expand
# ISO datetime formate
df['time'] = pd.to_datetime(df['time'])

# change 'variables' into columns, and 'value' are the value of columns
df_tmp = df.pivot(columns='variable', values='value')
df_expand = pd.concat([df[['id', 'time']].reset_index(drop=True), df_tmp.reset_index(drop=True)], axis=1)
df_expand.to_csv('../input/df_expand.csv', index=False)

trimmed_df = trim_outliers(df_expand)

# 1C
df_agg = pd.read_csv("../input/df_agg_hour.csv")
df_fe = df_agg[['id', 'day', 'hour', 'activity', 'mood', 'screen', 'circumplex.arousal', 'circumplex.valence']].copy()
df_fe["positive_app_time"] = df_agg["appCat.entertainment"] + df_agg["appCat.game"] + df_agg["appCat.travel"] + df_agg["appCat.social"]
df_fe["neutral_app_time"] = df_agg["appCat.builtin"] + df_agg["appCat.communication"] + df_agg["appCat.finance"] + df_agg["appCat.other"] + df_agg["appCat.unknown"] + df_agg["appCat.utilities"] + df_agg["appCat.weather"]
df_fe["negative_app_time"] = df_agg["appCat.office"]
df_fe["communications"] = df_agg["call"] + df_agg["sms"]

print(df_fe.loc[30:35,:])

df_rolling = df_fe[['id', 'day', 'hour', 'mood']].copy() # we need to find a way to do the below nicer
df_rolling[["activity", "screen", "circumplex.arousal", "circumplex.valence", "positive_app_time", "neutral_app_time", "negative_app_time", "communications"]] = df_fe[["activity", "screen", "circumplex.arousal", "circumplex.valence", "positive_app_time", "neutral_app_time", "negative_app_time", "communications"]].rolling(window=5, min_periods=1).mean()

print(df_rolling.loc[30:35,:])