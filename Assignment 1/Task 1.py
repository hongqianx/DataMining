import pandas as pd
from sklearn.neural_network import MLPClassifier

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



# ATTEMPT
# Neural networks have many options
# Works well for complex data.
#

