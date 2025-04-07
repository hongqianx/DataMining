import pandas as pd
import matplotlib.pyplot as plt
import math

def printMetric(desc, metric):
    print(desc + " is " + str(metric) + ".")

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

def box_plot(columns, fig_path):
    df_expand[columns].plot(kind='box', figsize=(8, 5), vert=False, patch_artist=True,
                                boxprops=dict(facecolor='lightblue', color='gray'),
                                medianprops=dict(color='red'),
                                flierprops=dict(marker='o', markerfacecolor='orange', markersize=5, linestyle='none'))

    plt.title("Box Plot")
    plt.xlabel("Value")
    plt.grid(True, axis='x', linestyle='--', alpha=0.6)
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
# ISO datetime formate
df['time'] = pd.to_datetime(df['time'])

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
# select numerical columns
value_cols = df_expand.select_dtypes(include='number').columns

trimmed_df = trim_outliers(df_expand)

# set image layout
num_vars = len(value_cols)
cols = 5  # images per column
rows = math.ceil(num_vars / cols)

# create canvas
fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
axes = axes.flatten()

# for each col
for i, col in enumerate(value_cols):
    axes[i].hist(df_expand[col].dropna(), bins=10, color='skyblue', edgecolor='black')
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')
    axes[i].set_yscale('log')

# hide empty image
for j in range(len(value_cols), len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.savefig("../image/origin_dist.png", dpi=500)
plt.clf()

# 3.1 Box plot
columns_to_plot1 = ['activity','circumplex.arousal', 'circumplex.valence','call','sms','mood']
fig_path1 = "../image/box_plot1.png"
columns_to_plot2 = ['appCat.builtin', 'appCat.communication','appCat.entertainment', 'appCat.finance', 'appCat.game',\
                    'appCat.office', 'appCat.other', 'appCat.social', 'appCat.travel','appCat.unknown', 'appCat.utilities',\
                    'appCat.weather', 'screen']
fig_path2 = "../image/box_plot2.png"

box_plot(columns_to_plot1,fig_path1)
box_plot(columns_to_plot2,fig_path2)

# 4. aggregate data by day
df_expand['day'] = df_expand['time'].dt.date
group_cols = ['id', 'day']

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
df_agg = df_expand.groupby(group_cols).agg(agg_dict).reset_index()

df_agg.to_csv('../input/df_agg.csv', index=False)

df_agg = pd.read_csv('../input/df_agg.csv')
# check missing value percentage
missing_ratio = df_agg.isna().mean()
# get value range
info_stat = df_agg.describe(include="all")

fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
axes = axes.flatten()

# for each col
for i, col in enumerate(value_cols):
    print(trimmed_df[col])
    axes[i].hist(trimmed_df[col].dropna(), bins=10, color='skyblue', edgecolor='black')
    axes[i].set_title(f'Distribution of trimmed {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')
    #axes[i].set_yscale('log')  # log scale for better visibility

# hide empty image
for j in range(len(value_cols), len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.savefig("../image/trimmed_dist.png", dpi=500)
plt.clf()