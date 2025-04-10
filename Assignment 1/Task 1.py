import numpy as np
import pandas as pd

# Import our data
input_data = r"../input/dataset_mood_smartphone.csv"
df = pd.read_csv(input_data)

# Clean our data
# for columns in data, change value <0, and upper 0.99 quantile as NA
def trim_outliers(data, columns, upper_percent=0.99):
    df_trim = data.copy()
    for col in columns:
        upper_bound = df_trim[col].quantile(upper_percent)
        df_trim[col] = df_trim[col].where((df_trim[col] >= 0) & (df_trim[col] <= upper_bound), np.nan)
    return df_trim

df.columns.tolist()

# Time value correlation
df_with_epoch = df.assign(epoch_time=(pd.to_datetime(df['time']) - pd.Timestamp('1970-01-01')) / pd.Timedelta('1s'))
print(df_with_epoch['value'].corr(df_with_epoch['epoch_time']))

# 2. create df_expand, input: df, output: df_expand
# ISO datetime formate
df['time'] = pd.to_datetime(df['time'])

#Convert user to numeric tag
df['id'] = df['id'].astype('category').cat.codes
df['time_bin'] = df['time'].astype('category').cat.codes

# change 'variables' into columns, and 'value' are the value of columns
df_tmp = df.pivot(columns='variable', values='value')
df_expand = pd.concat([df[['id', 'time']].reset_index(drop=True), df_tmp.reset_index(drop=True)], axis=1)
df_expand.to_csv('../input/df_expand.csv', index=False)

# change outliers to NA, input: df_expand, output: df_trim
outlier_col = ['screen','appCat.builtin','appCat.communication',\
           'appCat.entertainment', 'appCat.finance', 'appCat.game',\
           'appCat.office', 'appCat.other', 'appCat.social', 'appCat.travel',\
           'appCat.unknown', 'appCat.utilities', 'appCat.weather']
df_trim = trim_outliers(data = df_expand, columns=outlier_col)
df_trim.to_csv('../input/df_trim.csv', index=False)

# 1C
df_agg = pd.read_csv("../input/df_agg_6hour.csv")
df_agg['id'] = df_agg['id'].astype('category').cat.codes
df_agg['time_bin'] = df_agg['time_bin'].astype('category').cat.codes

# Impute using linear interpolation
cols_to_interp = ['circumplex.valence', 'circumplex.arousal', 'activity']
before_interp = df_agg[cols_to_interp].isna().sum()
df_agg[cols_to_interp] = df_agg[cols_to_interp].interpolate(method='linear').fillna(method='bfill').round(3)
after_interp = df_agg[cols_to_interp].isna().sum()
df_agg.to_csv('../input/test_interp.csv', index=False)

print("Before interpolation:")
print(before_interp)
print("\nAfter interpolation:")
print(after_interp)

print(df_agg["mood"].isna().sum())
df_fe = df_agg[['id', 'time_bin', 'activity', 'mood', 'screen', 'circumplex.arousal', 'circumplex.valence']].copy()
df_fe["positive_app_time"] = df_agg["appCat.entertainment"] + df_agg["appCat.game"] + df_agg["appCat.travel"] + df_agg["appCat.social"]
df_fe["neutral_app_time"] = df_agg["appCat.builtin"] + df_agg["appCat.communication"] + df_agg["appCat.finance"] + df_agg["appCat.other"] + df_agg["appCat.unknown"] + df_agg["appCat.utilities"] + df_agg["appCat.weather"]
df_fe["negative_app_time"] = df_agg["appCat.office"]
df_fe["communications"] = df_agg["call"] + df_agg["sms"]

print(df_fe.loc[30:35,:])

df_rolling = df_fe[['id', 'time_bin', 'mood']].copy() # we need to find a way to do the below nicer
df_rolling[["activity", "screen", "circumplex.arousal", "circumplex.valence", "positive_app_time", "neutral_app_time", "negative_app_time", "communications"]] = df_fe[["activity", "screen", "circumplex.arousal", "circumplex.valence", "positive_app_time", "neutral_app_time", "negative_app_time", "communications"]].rolling(window=5, min_periods=1).mean()

print(df_rolling.loc[30:35,:])

