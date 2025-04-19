import numpy as np
import pandas as pd
from dataPrepare import DataPrepare
data_proc = DataPrepare()

# 1A
# 1.read data
input_data = r"../input/dataset_mood_smartphone.csv"
df = pd.read_csv(input_data)

# ISO datetime formate
df['time'] = pd.to_datetime(df['time'])


# 376912 records
data_proc.printMetric("Amount of records", len(df))

# NA ratio is 0.0107%
data_proc.printMetric("Ratio of NA values", f"{df.isna().mean().mean() * 100:.4f}%")

# Duplicate records is 0
data_proc.printMetric("Amount of duplicate records", df.duplicated().sum())

# Value mean, min and max
data_proc.stat_col(data=df, column='value')

# 27 unique users
data_proc.printMetric("Amount of unique users", df["id"].nunique())

# Time value correlation
df_with_epoch = df.assign(epoch_time=(pd.to_datetime(df['time']) - pd.Timestamp('1970-01-01')) / pd.Timedelta('1s'))
print(df_with_epoch['value'].corr(df_with_epoch['epoch_time']))


# change 'variables' into columns, and 'value' are the value of columns
df_expand = data_proc.data_pivot(data=df)

# image distribution for all columns
data_proc.hist_plot(column = df_expand['mood'], fig_path = "../image/mood.png")
data_proc.hist_plot(column = df_expand['activity'], fig_path = "../image/activity.png")

# Box plot
columns_to_plot1 = ['activity','circumplex.arousal', 'circumplex.valence','call','sms','mood']
fig_path1 = "../image/box_plot1.png"
columns_to_plot2 = ['appCat.builtin', 'appCat.communication','appCat.entertainment', 'appCat.finance', 'appCat.game',\
                    'appCat.office', 'appCat.other', 'appCat.social', 'appCat.travel','appCat.unknown', 'appCat.utilities',\
                    'appCat.weather', 'screen']
fig_path2 = "../image/box_plot2.png"

data_proc.box_plot(df_expand, columns_to_plot1,fig_path1)
data_proc.box_plot(df_expand, columns_to_plot2,fig_path2)


# 1B
# process outliers
df_trim = data_proc.outlier_process(data=df_expand)
# plot hist for numerical columns
value_cols = df_trim.select_dtypes(include='number').columns
data_proc.multi_hist_plot(columns=value_cols, fig_path="../image/trimmed_dist.png",data=df_trim )


# aggregate data
df_agg = data_proc.data_aggregate(data=df_trim)
# check missing value percentage
print(df_agg.isna().mean())

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


