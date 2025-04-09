import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Load raw data from current directory
df_raw = pd.read_csv("../input/dataset_mood_smartphone.csv")
df_raw['time'] = pd.to_datetime(df_raw['time'])

# Filter mood values
df_mood = df_raw[df_raw['variable'] == 'mood']
user_id = "AS14.01"
df_user = df_mood[df_mood['id'] == user_id].sort_values('time')

# Convert time to numeric format for regression
df_user['time_num'] = df_user['time'].map(pd.Timestamp.toordinal)

# Fit linear regression (one straight trend line)
slope, intercept, r_value, p_value, std_err = linregress(df_user['time_num'], df_user['value'])
df_user['trend'] = intercept + slope * df_user['time_num']

# Plot mood values and linear trend line
plt.figure(figsize=(12, 5))
plt.plot(df_user['time'], df_user['value'], marker='o', linestyle='-', label='Mood')
plt.plot(df_user['time'], df_user['trend'], color='red', linewidth=2, label='Linear Trend Line')
plt.title(f'Mood Over Time with Linear Trend — User {user_id}')
plt.xlabel('Time')
plt.ylabel('Mood')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
