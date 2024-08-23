import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sb


def summary_statistics(df):
    return df.describe()
def data_quality_check(df):
    missing_values = df.isnull().sum()
    negative_values = (df[['GHI', 'DNI', 'DHI']] < 0).sum()
    return missing_values


def outliers_fun(df, cols):    
    outliers_arr = []
    threshold = 3
    
    df_copy = df.copy()
    
    for col in cols:
        # Calculate Z-scores for the entire column
        z_scores = stats.zscore(df_copy[col].dropna())
        outlier_mask = (abs(z_scores) > threshold)
        print("----------------mask------------",outlier_mask[674:])
        
        # Collect outliers
        outliers = df_copy[col].dropna()[outlier_mask]
        # print("--------outlier-----------",outliers[674:])
        outliers_arr.append(outliers)
        
        # Replace outliers with mean
        mean_value = df_copy[col].dropna()[~outlier_mask].mean()
        # print("--------mean-----------",mean_value)

        
        # Apply replacement based on the Z-scores in the copy
        df_copy[col] = np.where(outlier_mask, mean_value, df_copy[col])
        # print(df_copy[col])
    
    return df_copy

def line_graph_over_a_day(df):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    filtered_df = df[df['Timestamp'].dt.strftime('%Y-%m-%d') == '2022-01-01']
    filtered_df.set_index('Timestamp', inplace=True)
    plt.figure(figsize=(10, 6))

    plt.plot(filtered_df.index, filtered_df['GHI'], label='GHI', marker='o')
    plt.plot(filtered_df.index, filtered_df['DNI'], label='DNI', marker='o')
    plt.plot(filtered_df.index, filtered_df['DHI'], label='DHI', marker='o')
    plt.plot(filtered_df.index, filtered_df['Tamb'], label='Tamb', marker='o')

    plt.title('GHI, DNI, and DHI over a Day')
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
def line_graph_over_a_month(df):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    filtered_df = df[df['Timestamp'].dt.strftime('%Y-%m') == '2022-01']
    filtered_df.set_index('Timestamp', inplace=True)
    plt.figure(figsize=(10, 6))

    plt.plot(filtered_df.index, filtered_df['GHI'], label='GHI', marker='o')
    plt.plot(filtered_df.index, filtered_df['DNI'], label='DNI', marker='o')
    plt.plot(filtered_df.index, filtered_df['DHI'], label='DHI', marker='o')
    plt.plot(filtered_df.index, filtered_df['Tamb'], label='Tamb', marker='o')

    plt.title('GHI, DNI, and DHI over a Month')
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
def correlation(df):
    df_correlation=df[['GHI','DNI','DHI','TModA', 'TModB']].corr()
    plt.figure(figsize=(10, 6))
    sb.heatmap(df_correlation,annot=True)
    sb.pairplot(df_correlation)
    print("1 2 3 4 5 6 7 8 9 10 11 yyyyyyhkhhhjhj---------------------")
    # plt.show()
    
    return df_correlation