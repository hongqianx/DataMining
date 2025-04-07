import pandas as pd

# Import our data
input_data = r"../input/dataset_mood_smartphone.csv"
df = pd.read_csv(input_data)

# Clean our data
# Select the 0.1st until 99.9 quantile of our data. (581 lost)
df_filtered = df[(df['value'] >= df['value'].quantile(0.001)) & (df['value'] <= df['value'].quantile(0.999))]