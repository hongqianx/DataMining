import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Load data
df_raw = pd.read_csv("../input/dataset_mood_smartphone.csv")
df_raw['time'] = pd.to_datetime(df_raw['time'])

# Filter only mood entries
df_mood = df_raw[df_raw['variable'] == 'mood'].sort_values('time')

# Convert time to ordinal (numeric) for regression
df_mood['time_num'] = df_mood['time'].map(pd.Timestamp.toordinal)

# Linear regression across all users
slope, intercept, r_value, p_value, std_err = linregress(df_mood['time_num'], df_mood['value'])
df_mood['trend'] = intercept + slope * df_mood['time_num']

# Plot all mood points and trend line
plt.figure(figsize=(14, 6))
plt.scatter(df_mood['time'], df_mood['value'], alpha=0.3, label='All Mood Points', s=10)
plt.plot(df_mood['time'], df_mood['trend'], color='red', linewidth=2.5, label='Overall Linear Trend')
plt.title("Overall Mood Trend Across All Users")
plt.xlabel("Time")
plt.ylabel("Mood")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
