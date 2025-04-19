import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np

class DataPrepare:
    def printMetric(self, desc, metric):
        print(desc + " is " + str(metric) + ".")

# for columns in data, change value <0, and upper 0.99 quantile as NA
    def trim_outliers(self, data, columns, upper_percent=0.99):
        df_trim = data.copy()
        for col in columns:
            upper_bound = df_trim[col].quantile(upper_percent)
            df_trim[col] = df_trim[col].where((df_trim[col] >= 0) & (df_trim[col] <= upper_bound), np.nan)
        return df_trim

    # Value mean, min and max
    def stat_col(self, data, column):
        mean_value = data[column].mean()
        min_value = data[column].min()
        max_value = data[column].max()
        median_value = data[column].median()
        variance_value = data[column].var()
        self.printMetric("Mean value", mean_value)
        self.printMetric("Minimum value", min_value)
        self.printMetric("Maximum value", max_value)
        self.printMetric("Median value", median_value)
        self.printMetric("Variance value", variance_value)


    # draw one box plot for multiple columns
    def box_plot(self, data, columns, fig_path):
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
    def hist_plot(self, column, fig_path):
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
    def multi_hist_plot(self, columns, fig_path, data):
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

    def compare_stat(self, summary_before, summary_after):
        comparison = summary_before.T.join(summary_after.T, lsuffix='_before', rsuffix='_after')
        print(comparison[['mean_before', 'mean_after', 'std_before', 'std_after', 'min_before', 'min_after', 'max_before', 'max_after']])
        return comparison

    def outlier_process(self, data):
        # for ['mood','circumplex.arousal','circumplex.valence','activity','sms','call'], outlier is point outside of range in definition
        count_outside = data[(data['mood'] < 1) | (data['mood'] > 10)].shape[0]
        print('mood has ' + str(count_outside) + ' outliers')
        count_outside = data[(data['circumplex.arousal'] < -2) | (data['circumplex.arousal'] > 2)].shape[0]
        print('circumplex.arousal has ' + str(count_outside) + ' outliers')
        count_outside = data[(data['circumplex.valence'] < -2) | (data['circumplex.valence'] > 2)].shape[0]
        print('circumplex.valence has ' + str(count_outside) + ' outliers')
        count_outside = data[(data['activity'] < 0) | (data['activity'] > 1)].shape[0]
        print('activity has ' + str(count_outside) + ' outliers')
        count_outside = data[~((data['sms'] == 1) | (data['sms'].isna()))].shape[0]
        print('sms has ' + str(count_outside) + ' outliers')
        count_outside = data[~((data['call'] == 1) | (data['call'].isna()))].shape[0]
        print('call has ' + str(count_outside) + ' outliers')

        # for other columns, we delete value <0, and upper 0.99 quantile
        outlier_col = ['screen','appCat.builtin','appCat.communication',\
                   'appCat.entertainment', 'appCat.finance', 'appCat.game',\
                   'appCat.office', 'appCat.other', 'appCat.social', 'appCat.travel',\
                   'appCat.unknown', 'appCat.utilities', 'appCat.weather']


        df_trim = self.trim_outliers(data = data, columns=outlier_col)
        return df_trim

    def data_pivot(self, data):
        df_tmp = data.pivot(columns='variable', values='value')
        df_expand = pd.concat([df[['id', 'time']].reset_index(drop=True), df_tmp.reset_index(drop=True)], axis=1)
        return df_expand

    def data_aggregate(self, data):
        data['time_bin'] = data['time'].dt.floor(f'{6}H')
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
        df_agg = data.groupby(group_cols).agg(agg_dict).reset_index()
        return df_agg
