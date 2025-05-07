import logging
import pandas as pd
import matplotlib.pyplot as plt


# Set up a basic logger
logger = logging.getLogger("MLLogger")
logger.setLevel(logging.DEBUG)  # Set the global logging level

# Import dataset
input_data = r"../input/training_set_VU_DM.csv"
df = pd.read_csv(input_data)

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

def compare_stat(summary_before, summary_after):
    comparison = summary_before.T.join(summary_after.T, lsuffix='_before', rsuffix='_after')
    print(comparison[['mean_before', 'mean_after', 'std_before', 'std_after', 'min_before', 'min_after', 'max_before', 'max_after']])
    return comparison


# x records
printMetric("Amount of records", len(df))

# NA ratio is x
printMetric("Ratio of NA values", f"{df.isna().mean().mean() * 100:.4f}%")

# Duplicate records is x
printMetric("Amount of duplicate records", df.duplicated().sum())




# Plots