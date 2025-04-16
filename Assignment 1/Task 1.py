import numpy as np
import pandas as pd
from dataPrepare import DataPrepare

# 1A
# 1.read data
input_data = r"../input/dataset_mood_smartphone.csv"
df = pd.read_csv(input_data)
data_proc = DataPrepare()
# ISO datetime formate
df['time'] = pd.to_datetime(df['time'])

# change 'variables' into columns, and 'value' are the value of columns
df_expand = data_proc.data_pivot(data=df)

# 1B
# process outliers
df_trim = data_proc.outlier_process(data=df_expand)
# aggregate data
df_agg = data_proc.data_aggregate(data=df_trim)

# 1C
df_agg['id'] = df_agg['id'].astype('category').cat.codes
df_agg['time_bin'] = df_agg['time_bin'].astype('category').cat.codes

# Impute using linear interpolation
cols_to_interp = ['circumplex.valence', 'circumplex.arousal', 'activity', 'mood']
before_interp = df_agg[cols_to_interp].isna().sum()
df_agg[cols_to_interp] = df_agg[cols_to_interp].interpolate(method='linear').fillna(method='bfill').round(3)
after_interp = df_agg[cols_to_interp].isna().sum()
df_agg.to_csv('../input/df_interp_6hour.csv', index=False)

# Every column containing an NA value needs to be interpolated, everything except ID, Time bin and Mood.

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

df_rolling.to_csv('../input/df_rolling.csv', index=False)

print(df_rolling.loc[30:35,:])


