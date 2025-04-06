import pandas as pd
import matplotlib.pyplot as plt
import math

# 1.read data
input_data = r"input/dataset_mood_smartphone.csv"
df = pd.read_csv(input_data)
# 376912 records
len(df)
# column =  ['Unnamed: 0', 'id', 'time', 'variable', 'value']
df.columns.tolist()
# 27 unique users
df["id"].nunique()

# 2. create df_expand
# ISO datetime formate
df['time'] = pd.to_datetime(df['time'])

# change 'variables' into columns, and 'value' are the value of columns
df_tmp = df.pivot(columns='variable', values='value')
df_expand = pd.concat([df[['id', 'time']].reset_index(drop=True), df_tmp.reset_index(drop=True)], axis=1)
df_expand.to_csv('input/df_expand.csv', index=False)

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

# hide empty image
for j in range(len(value_cols), len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.savefig("image/origin_dist.png", dpi=500)


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

df_agg.to_csv('input/df_agg.csv', index=False)