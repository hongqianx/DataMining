import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np

def printMetric(desc, metric):
    print(desc + " is " + str(metric) + ".")

# for columns in data, change value <0, and upper 0.99 quantile as NA
def trim_outliers(data, columns, upper_percent=0.99):
    df_trim = data.copy()
    for col in columns:
        upper_bound = df_trim[col].quantile(upper_percent)
        df_trim[col] = df_trim[col].where((df_trim[col] >= 0) & (df_trim[col] <= upper_bound), np.nan)
    return df_trim



# draw one box plot for multiple columns
def box_plot(data, columns, fig_path):
    data[columns].plot(kind='box', figsize=(8, 5), vert=False, patch_artist=True,
                                boxprops=dict(facecolor='lightblue', color='gray'),
                                medianprops=dict(color='red'),
                                flierprops=dict(marker='o', markerfacecolor='orange', markersize=5, linestyle='none'))

    plt.title("Box Plot")
    plt.xlabel("Value")
    plt.grid(True, axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=500)
    plt.clf()

# draw single histogram for column
def hist_plot(column, fig_path):
    plt.figure(figsize=(8, 5))
    plt.hist(column.dropna(), bins=30, color='skyblue', edgecolor='black')
    plt.title('Histogram')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.clf()

# draw multiple histogram for columns
def multi_hist_plot(columns, fig_path, data):
    # set image layout
    num_vars = len(columns)
    cols = 5  # images per column
    rows = math.ceil(num_vars / cols)

    # create canvas
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = axes.flatten()

    # for each col
    for i, col in enumerate(columns):
        axes[i].hist(data[col].dropna(), bins=10, color='skyblue', edgecolor='black')
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
        # axes[i].set_yscale('log') # log scale for better visibility

    # hide empty image
    for j in range(len(columns), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(fig_path, dpi=500)
    plt.clf()

# 1.read data
input_data = r"../input/dataset_mood_smartphone.csv"
df = pd.read_csv(input_data)

# 376912 records
printMetric("Amount of records", len(df))

# NA ratio is 0.0107%
printMetric("Ratio of NA values", f"{df.isna().mean().mean() * 100:.4f}%")

# Duplicate records is 0
printMetric("Amount of duplicate records", df.duplicated().sum())

# Timeframe range is from 2014-02-17 07:00 to 2014-06-09 00:00
# ISO datetime formate
df['time'] = pd.to_datetime(df['time'])
min_time = df['time'].min()
max_time = df['time'].max()
printMetric("Minimum time", min_time)
printMetric("Maximum time", max_time)

# Value mean, min and max
mean_value = df['value'].mean()
min_value = df['value'].min()
max_value = df['value'].max()
median_value = df['value'].median()
variance_value = df['value'].var()
printMetric("Mean value", mean_value)
printMetric("Minimum value", min_value)
printMetric("Maximum value", max_value)
printMetric("Median value", median_value)
printMetric("Variance value", variance_value)

# Amount of dataset columns is 5
printMetric("Dataset columns", df.shape[1])

# column =  ['Unnamed: 0', 'id', 'time', 'variable', 'value']
df.columns.tolist()

# 27 unique users
printMetric("Amount of unique users", df["id"].nunique())

# Time value correlation
df_with_epoch = df.assign(epoch_time=(pd.to_datetime(df['time']) - pd.Timestamp('1970-01-01')) / pd.Timedelta('1s'))
print(df_with_epoch['value'].corr(df_with_epoch['epoch_time']))

# 2. create df_expand
# change 'variables' into columns, and 'value' are the value of columns
df_tmp = df.pivot(columns='variable', values='value')
df_expand = pd.concat([df[['id', 'time']].reset_index(drop=True), df_tmp.reset_index(drop=True)], axis=1)
df_expand.to_csv('../input/df_expand.csv', index=False)

# get info
# info: 21 columns, # non-null count, datatype
df_expand.info()
# statistics
info_stat = df_expand.describe(include="all")

# Checking missing values
print(df_expand.isnull().sum())

# Unique values per column
info_uni = df_expand.nunique()

# 3. image distribution for all columns
hist_plot(column = df_expand['mood'], fig_path = "../image/mood.png")
hist_plot(column = df_expand['activity'], fig_path = "../image/activity.png")


# 3.1 Box plot
columns_to_plot1 = ['activity','circumplex.arousal', 'circumplex.valence','call','sms','mood']
fig_path1 = "../image/box_plot1.png"
columns_to_plot2 = ['appCat.builtin', 'appCat.communication','appCat.entertainment', 'appCat.finance', 'appCat.game',\
                    'appCat.office', 'appCat.other', 'appCat.social', 'appCat.travel','appCat.unknown', 'appCat.utilities',\
                    'appCat.weather', 'screen']
fig_path2 = "../image/box_plot2.png"

box_plot(df_expand, columns_to_plot1,fig_path1)
box_plot(df_expand, columns_to_plot2,fig_path2)

# 4. outlier
# for ['mood','circumplex.arousal','circumplex.valence','activity','sms','call'], outlier is point outside of range in definition
count_outside = df_expand[(df_expand['mood'] < 1) | (df_expand['mood'] > 10)].shape[0]
print('mood has ' + str(count_outside) + ' outliers')
count_outside = df_expand[(df_expand['circumplex.arousal'] < -2) | (df_expand['circumplex.arousal'] > 2)].shape[0]
print('circumplex.arousal has ' + str(count_outside) + ' outliers')
count_outside = df_expand[(df_expand['circumplex.valence'] < -2) | (df_expand['circumplex.valence'] > 2)].shape[0]
print('circumplex.valence has ' + str(count_outside) + ' outliers')
count_outside = df_expand[(df_expand['activity'] < 0) | (df_expand['activity'] > 1)].shape[0]
print('activity has ' + str(count_outside) + ' outliers')
count_outside = df_expand[~((df_expand['sms'] == 1) | (df_expand['sms'].isna()))].shape[0]
print('sms has ' + str(count_outside) + ' outliers')
count_outside = df_expand[~((df_expand['call'] == 1) | (df_expand['call'].isna()))].shape[0]
print('call has ' + str(count_outside) + ' outliers')

# for other columns, we delete value <0, and upper 0.99 quantile
outlier_col = ['screen','appCat.builtin','appCat.communication',\
           'appCat.entertainment', 'appCat.finance', 'appCat.game',\
           'appCat.office', 'appCat.other', 'appCat.social', 'appCat.travel',\
           'appCat.unknown', 'appCat.utilities', 'appCat.weather']


df_trim = trim_outliers(data = df_expand, columns=outlier_col)
# get value range
info_stat = df_trim.describe(include="all")
df_trim.to_csv('../input/df_trim.csv', index=False)

# select numerical columns
value_cols = df_trim.select_dtypes(include='number').columns

multi_hist_plot(columns=value_cols, fig_path="../image/trimmed_dist.png",data=df_trim )


# 5. aggregate data by day
print(df_trim['time'].head())
# df_expand['day'] = df_expand['time'].dt.date
# df_expand['hour'] = df_expand['time'].dt.hour
# group_cols = ['id', 'day', 'hour']

df_trim['time_bin'] = df_trim['time'].dt.floor(f'{6}H')
group_cols = ['id', 'time_bin']

# define aggregate function
agg_dict = {
    'activity': 'mean',
    'appCat.builtin': 'sum',
    'appCat.communication': 'sum',
    'appCat.entertainment': 'sum',
    'appCat.finance': 'sum',
    'appCat.game': 'sum',
    'appCat.office': 'sum',
    'appCat.other': 'sum',
    'appCat.social': 'sum',
    'appCat.travel': 'sum',
    'appCat.unknown': 'sum',
    'appCat.utilities': 'sum',
    'appCat.weather': 'sum',
    'call': 'sum',
    'circumplex.arousal': 'mean',
    'circumplex.valence': 'mean',
    'mood': 'mean',
    'screen': 'sum',
    'sms': 'sum'
}

# aggregate data by id and day
df_agg = df_trim.groupby(group_cols).agg(agg_dict).reset_index()

df_agg.to_csv('../input/df_agg_6hour.csv', index=False)

# df_agg = pd.read_csv('../input/df_agg_hour.csv')
# check missing value percentage
missing_ratio = df_agg.isna().mean()
# get value range
info_stat = df_agg.describe(include="all")
