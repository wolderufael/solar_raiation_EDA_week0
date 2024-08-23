import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sb


def check_missing_value(df):
    missing_values = df.isnull().sum()
    # negative_values = (df[['GHI', 'DNI', 'DHI']] < 0).sum()
    return missing_values

def replace_nagative_values(df):
     df_copy = df.copy()
     df_copy[['GHI', 'DNI', 'DHI']] = df_copy[['GHI', 'DNI', 'DHI']].clip(lower=0)
     return df_copy

def outliers_fun(df, cols):    
    outliers_arr = []
    threshold = 3
    
    df_copy = df.copy()
    
    for col in cols:
        # Calculate Z-scores for the entire column
        z_scores = stats.zscore(df_copy[col].dropna())
        outlier_mask = (abs(z_scores) > threshold)
        # print("----------------mask------------",outlier_mask[674:])
        
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

def summary_statistics(df):
    return df.describe()

def mean_median_std_comparison(df1,df2,df3):
    #mean comparison plot
    mean_ghi_1 = df1['GHI'].mean()
    mean_ghi_2 = df2['GHI'].mean()
    mean_ghi_3 = df3['GHI'].mean()
    
    data = {
    'Site': ['Benin', 'Sierraleone', 'Togo'],
    'Mean GHI': [mean_ghi_1, mean_ghi_2, mean_ghi_3]
    }
    mean_df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    bars= plt.bar(mean_df['Site'], mean_df['Mean GHI'],color=['blue', 'green', 'orange'])
    
    plt.bar_label(bars, fmt='%.2f')

    
    plt.title('Comparison of Mean GHI Across Three Sites')
    plt.xlabel('Site')
    plt.ylabel('Mean GHI')
    # plt.legend()
    plt.grid(True)
    plt.show()
    #median comparison plot
    median_ghi_1 = df1['GHI'].median()
    median_ghi_2 = df2['GHI'].median()
    median_ghi_3 = df3['GHI'].median()
    
    data = {
    'Site': ['Benin', 'Sierraleone', 'Togo'],
    'Mean GHI': [median_ghi_1, median_ghi_2, median_ghi_3]
    }
    median_df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    plt.bar(median_df['Site'], median_df['Mean GHI'], color=['blue', 'green', 'orange'])

    plt.title('Comparison of Median GHI Across Three Sites')
    plt.xlabel('Site')
    plt.ylabel('Mean GHI')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    std_ghi_1 = df1['GHI'].std()
    std_ghi_2 = df2['GHI'].std()
    std_ghi_3 = df3['GHI'].std()
    
    data = {
    'Site': ['Benin', 'Sierraleone', 'Togo'],
    'Mean GHI': [std_ghi_1 , std_ghi_2 , std_ghi_3 ]
    }
    std_df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    plt.bar(std_df['Site'], std_df['Mean GHI'], color=['blue', 'green', 'orange'])

    plt.title('Comparison of Standard Deviation of GHI Across Three Sites')
    plt.xlabel('Site')
    plt.ylabel('Mean GHI')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
    
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
    plt.show()
    
    return df_correlation