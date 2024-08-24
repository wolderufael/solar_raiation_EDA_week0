import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sb
from functools import reduce


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
    #mean comparison plot of GHI
    meann_DNI_1 = df1['GHI'].mean()
    mean_ghi_2 = df2['GHI'].mean()
    mean_ghi_3 = df3['GHI'].mean()
    
    data = {
    'Site': ['Benin', 'Sierraleone', 'Togo'],
    'Mean GHI': [meann_DNI_1, mean_ghi_2, mean_ghi_3]
    }
    mean_df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    bars= plt.bar(mean_df['Site'], mean_df['Mean GHI'],color=['blue', 'green', 'orange'])
    
    plt.bar_label(bars, fmt='%.2f')

    
    plt.title('Comparison of Mean GHI Across Three Sites')
    plt.xlabel('Site')
    plt.ylabel('Mean GHI')
    plt.show()
    #median comparison plot of GHI
    median_ghi_1 = df1['GHI'].median()
    median_ghi_2 = df2['GHI'].median()
    median_ghi_3 = df3['GHI'].median()
    
    data = {
    'Site': ['Benin', 'Sierraleone', 'Togo'],
    'Median GHI': [median_ghi_1, median_ghi_2, median_ghi_3]
    }
    median_df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    bars=plt.bar(median_df['Site'], median_df['Median GHI'], color=['blue', 'green', 'orange'])
    
    plt.bar_label(bars, fmt='%.2f')

    plt.title('Comparison of Median GHI Across Three Sites')
    plt.xlabel('Site')
    plt.ylabel('Median GHI')
    plt.show()
    #standard deviation comparison plot of GHI
    std_ghi_1 = df1['GHI'].std()
    std_ghi_2 = df2['GHI'].std()
    std_ghi_3 = df3['GHI'].std()
    
    data = {
    'Site': ['Benin', 'Sierraleone', 'Togo'],
    'Std GHI': [std_ghi_1 , std_ghi_2 , std_ghi_3 ]
    }
    std_df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    bars=plt.bar(std_df['Site'], std_df['Std GHI'], color=['blue', 'green', 'orange'])
    
    plt.bar_label(bars, fmt='%.2f')

    plt.title('Comparison of Standard Deviation of GHI Across Three Sites')
    plt.xlabel('Site')
    plt.ylabel('Std GHI')
    plt.show()
    
    ###########################################
        #mean comparison plot of DNI
    mean_dni_1 = df1['DNI'].mean()
    mean_dni_2 = df2['DNI'].mean()
    mean_dni_3 = df3['DNI'].mean()
    
    data = {
    'Site': ['Benin', 'Sierraleone', 'Togo'],
    'Mean DNI': [mean_dni_1, mean_dni_2, mean_dni_3]
    }
    mean_df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    bars= plt.bar(mean_df['Site'], mean_df['Mean DNI'],color=['blue', 'green', 'orange'])
    
    plt.bar_label(bars, fmt='%.2f')

    
    plt.title('Comparison of Mean DNI Across Three Sites')
    plt.xlabel('Site')
    plt.ylabel('Mean DNI')
    plt.show()
    #median comparison plot of DNI
    median_dni_1 = df1['DNI'].median()
    median_dni_2 = df2['DNI'].median()
    median_dni_3 = df3['DNI'].median()
    
    data = {
    'Site': ['Benin', 'Sierraleone', 'Togo'],
    'Median DNI': [median_dni_1, median_dni_2, median_dni_3]
    }
    median_df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    bars=plt.bar(median_df['Site'], median_df['Median DNI'], color=['blue', 'green', 'orange'])
    
    plt.bar_label(bars, fmt='%.2f')

    plt.title('Comparison of Median DNI Across Three Sites')
    plt.xlabel('Site')
    plt.ylabel('Median DNI')
    plt.show()
    #standard deviation comparison plot of DNI
    std_dni_1 = df1['DNI'].std()
    std_dni_2 = df2['DNI'].std()
    std_dni_3 = df3['DNI'].std()
    
    data = {
    'Site': ['Benin', 'Sierraleone', 'Togo'],
    'Std DNI': [std_dni_1 , std_dni_2 , std_dni_3 ]
    }
    std_df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    bars=plt.bar(std_df['Site'], std_df['Std DNI'], color=['blue', 'green', 'orange'])
    
    plt.bar_label(bars, fmt='%.2f')

    plt.title('Comparison of Standard Deviation of DNI Across Three Sites')
    plt.xlabel('Site')
    plt.ylabel('Std DNI')
    plt.show()
    ###########################################
    ###########################################
    #mean comparison plot of DHI
    mean_dhi_1 = df1['DHI'].mean()
    mean_dhi_2 = df2['DHI'].mean()
    mean_dhi_3 = df3['DHI'].mean()
    
    data = {
    'Site': ['Benin', 'Sierraleone', 'Togo'],
    'Mean DHI': [mean_dhi_1, mean_dhi_2, mean_dhi_3]
    }
    mean_df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    bars= plt.bar(mean_df['Site'], mean_df['Mean DHI'],color=['blue', 'green', 'orange'])
    
    plt.bar_label(bars, fmt='%.2f')

    
    plt.title('Comparison of Mean DHI Across Three Sites')
    plt.xlabel('Site')
    plt.ylabel('Mean DHI')
    plt.show()
    #median comparison plot of DHI
    median_dhi_1 = df1['DHI'].median()
    median_dhi_2 = df2['DHI'].median()
    median_dhi_3 = df3['DHI'].median()
    
    data = {
    'Site': ['Benin', 'Sierraleone', 'Togo'],
    'Median DHI': [median_dhi_1, median_dhi_2, median_dhi_3]
    }
    median_df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    bars=plt.bar(median_df['Site'], median_df['Median DHI'], color=['blue', 'green', 'orange'])
    
    plt.bar_label(bars, fmt='%.2f')

    plt.title('Comparison of Median DHI Across Three Sites')
    plt.xlabel('Site')
    plt.ylabel('Median DHI')
    plt.show()
    #standard deviation comparison plot of DHI
    std_dhi_1 = df1['DHI'].std()
    std_dhi_2 = df2['DHI'].std()
    std_dhi_3 = df3['DHI'].std()
    
    data = {
    'Site': ['Benin', 'Sierraleone', 'Togo'],
    'Std DHI': [std_dhi_1 , std_dhi_2 , std_dhi_3 ]
    }
    std_df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    bars=plt.bar(std_df['Site'], std_df['Std DHI'], color=['blue', 'green', 'orange'])
    
    plt.bar_label(bars, fmt='%.2f')

    plt.title('Comparison of Standard Deviation of DNI Across Three Sites')
    plt.xlabel('Site')
    plt.ylabel('Std DHI')
    plt.show()
    ########################
    
def plot_over_an_hour(df1, df2, df3):
    # Convert 'Timestamp' columns to datetime
    for df in [df1, df2, df3]:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Filter data for a specific date
    date_str = '2022-01-01 12'
    filtered_dfs = [df[df['Timestamp'].dt.strftime('%Y-%m-%d %H') == date_str] for df in [df1, df2, df3]]
    
    # Set 'Timestamp' as index for plotting
    for df in filtered_dfs:
        df.set_index('Timestamp', inplace=True)
    
    # Create subplots
    fig, axs = plt.subplots(4, 1, figsize=(14, 18))
    fig.suptitle('GHI, DNI, DHI, and Tamb over an Hour', fontsize=16)
    
    # Plot GHI
    for df, label in zip(filtered_dfs, ['Benin', 'Sierraleone', 'Togo']):
        axs[0].plot(df.index, df['GHI'], label=f'{label}', marker='o')
    axs[0].set_title('GHI')
    axs[0].set_ylabel('GHI')
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot DNI
    for df, label in zip(filtered_dfs, ['Benin', 'Sierraleone', 'Togo']):
        axs[1].plot(df.index, df['DNI'], label=f'{label}', marker='o')
    axs[1].set_title('DNI')
    axs[1].set_ylabel('DNI')
    axs[1].legend()
    axs[1].grid(True)
    
    # Plot DHI
    for df, label in zip(filtered_dfs, ['Benin', 'Sierraleone', 'Togo']):
        axs[2].plot(df.index, df['DHI'], label=f'{label}', marker='o')
    axs[2].set_title('DHI')
    axs[2].set_ylabel('DHI')
    axs[2].legend()
    axs[2].grid(True)
    
    # Plot Tamb
    for df, label in zip(filtered_dfs, ['Benin', 'Sierraleone', 'Togo']):
        axs[3].plot(df.index, df['Tamb'], label=f'{label}', marker='o')
    axs[3].set_title('Tamb')
    axs[3].set_ylabel('Tamb')
    axs[3].legend()
    axs[3].grid(True)
    
    # Formatting the x-axis
    axs[3].set_xlabel('Time')
    plt.xticks(rotation=45)
    plt.tight_layout(rect=[0, 0, 1, 0.97])  #
    
    plt.show()

def plot_over_a_day(df1, df2, df3):
    # Convert 'Timestamp' columns to datetime
    for df in [df1, df2, df3]:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Filter data for a specific date
    date_str = '2022-01-01'
    filtered_dfs = [df[df['Timestamp'].dt.strftime('%Y-%m-%d') == date_str] for df in [df1, df2, df3]]
    
    # Set 'Timestamp' as index for plotting
    for df in filtered_dfs:
        df.set_index('Timestamp', inplace=True)
    
    # Create subplots
    fig, axs = plt.subplots(4, 1, figsize=(14, 18))
    fig.suptitle('GHI, DNI, DHI, and Tamb over a Day', fontsize=16)
    
    # Plot GHI
    for df, label in zip(filtered_dfs, ['Benin', 'Sierraleone', 'Togo']):
        axs[0].plot(df.index, df['GHI'], label=f'{label}', marker='o')
    axs[0].set_title('GHI')
    axs[0].set_ylabel('GHI')
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot DNI
    for df, label in zip(filtered_dfs, ['Benin', 'Sierraleone', 'Togo']):
        axs[1].plot(df.index, df['DNI'], label=f'{label}', marker='o')
    axs[1].set_title('DNI')
    axs[1].set_ylabel('DNI')
    axs[1].legend()
    axs[1].grid(True)
    
    # Plot DHI
    for df, label in zip(filtered_dfs, ['Benin', 'Sierraleone', 'Togo']):
        axs[2].plot(df.index, df['DHI'], label=f'{label}', marker='o')
    axs[2].set_title('DHI')
    axs[2].set_ylabel('DHI')
    axs[2].legend()
    axs[2].grid(True)
    
    # Plot Tamb
    for df, label in zip(filtered_dfs, ['Benin', 'Sierraleone', 'Togo']):
        axs[3].plot(df.index, df['Tamb'], label=f'{label}', marker='o')
    axs[3].set_title('Tamb')
    axs[3].set_ylabel('Tamb')
    axs[3].legend()
    axs[3].grid(True)
    
    # Formatting the x-axis
    axs[3].set_xlabel('Time')
    plt.xticks(rotation=45)
    plt.tight_layout(rect=[0, 0, 1, 0.97])  #
    
    plt.show()

def plot_over_a_month(df1, df2, df3):
    # Convert 'Timestamp' columns to datetime
    for df in [df1, df2, df3]:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Filter data for a specific date
    date_str = '2022-01'
    filtered_dfs = [df[df['Timestamp'].dt.strftime('%Y-%m') == date_str] for df in [df1, df2, df3]]
    
    # Set 'Timestamp' as index for plotting
    for df in filtered_dfs:
        df.set_index('Timestamp', inplace=True)
    
    # Create subplots
    fig, axs = plt.subplots(4, 1, figsize=(14, 18))
    fig.suptitle('GHI, DNI, DHI, and Tamb over a Month', fontsize=16)
    
    # Plot GHI
    for df, label in zip(filtered_dfs, ['Benin', 'Sierraleone', 'Togo']):
        axs[0].plot(df.index, df['GHI'], label=f'{label}', marker='.')
    axs[0].set_title('GHI')
    axs[0].set_ylabel('GHI')
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot DNI
    for df, label in zip(filtered_dfs, ['Benin', 'Sierraleone', 'Togo']):
        axs[1].plot(df.index, df['DNI'], label=f'{label}', marker='.')
    axs[1].set_title('DNI')
    axs[1].set_ylabel('DNI')
    axs[1].legend()
    axs[1].grid(True)
    
    # Plot DHI
    for df, label in zip(filtered_dfs, ['Benin', 'Sierraleone', 'Togo']):
        axs[2].plot(df.index, df['DHI'], label=f'{label}', marker='.')
    axs[2].set_title('DHI')
    axs[2].set_ylabel('DHI')
    axs[2].legend()
    axs[2].grid(True)
    
    # Plot Tamb
    for df, label in zip(filtered_dfs, ['Benin', 'Sierraleone', 'Togo']):
        axs[3].plot(df.index, df['Tamb'], label=f'{label}', marker='.')
    axs[3].set_title('Tamb')
    axs[3].set_ylabel('Tamb')
    axs[3].legend()
    axs[3].grid(True)
    
    # Formatting the x-axis
    axs[3].set_xlabel('Time')
    plt.xticks(rotation=45)
    plt.tight_layout(rect=[0, 0, 1, 0.97])  #
    
    plt.show()

def merge_df(df1,df2,df3):
     dfs = [df1,df2,df3]
     merged_df = reduce(lambda left, right: pd.merge(left, right, on=['Timestamp', 'GHI', 'DNI', 'DHI', 'ModA', 'ModB', 'Tamb', 'RH', 'WS',
       'WSgust', 'WSstdev', 'WD', 'WDstdev', 'BP', 'Cleaning', 'Precipitation',
       'TModA', 'TModB'],how='outer'), dfs)
     return merged_df
    
def correlation(df):
    df_correlation_temp=df[['GHI','DNI','DHI','TModA', 'TModB']].corr()
    df_correlation_wind=df[['GHI','DNI','DHI','WS', 'WSgust', 'WD']].corr()
    
    plt.figure(figsize=(14, 12))
    
    plt.subplot(2, 1, 1)
    # First heatmap for temperature correlations
    plt.subplot(2, 1, 1)  # 2 rows, 1 column, position 1
    sb.heatmap(df_correlation_temp, annot=True, cmap='coolwarm')
    plt.title('Temperature and Solar Irradiance Correlation')
    
    # Second heatmap for wind correlations
    plt.subplot(2, 1, 2)  # 2 rows, 1 column, position 2
    sb.heatmap(df_correlation_wind, annot=True, cmap='coolwarm')
    plt.title('Wind Conditions and Solar Irradiance Correlation')

    plt.suptitle('Correlation Matrix of Temperature, Wind Conditions, and Solar Irradiance', y=1.02)
    plt.tight_layout()
    plt.show()    
    # return df_correlation
    
def polar_plot(df):
    df['WD_rad'] = np.deg2rad(df['WD'])

    # Wind speed data
    wind_speed = df['WS']
    wind_direction = df['WD_rad']
    # Create polar plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    # Histogram of wind speed
    hist, edges = np.histogram(df['WD_rad'], bins=36, range=(0, 2 * np.pi))
    width = np.diff(edges)
    bars = ax.bar(edges[:-1], hist, width=width, align='center', alpha=0.5, color='b')

    # Add wind speed to the plot
    for bar in bars:
        bar.set_facecolor(plt.cm.jet(np.random.rand()))  # Random color for each bar

    ax.set_title("Wind Speed and Direction Distribution")
    plt.show()
    # Plot wind direction variability
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot wind direction variability
    ax.hist(df['WDstdev'], bins=30, color='orange', edgecolor='black', alpha=0.7)

    ax.set_title("Wind Direction Variability")
    ax.set_xlabel("Wind Direction Standard Deviation")
    ax.set_ylabel("Frequency")
    plt.show()
def temperatur_analysis(df):
    corr_matrix = df[['RH', 'Tamb', 'GHI', 'DNI', 'DHI']].corr()
    print(corr_matrix)
    
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    sb.scatterplot(x='RH', y='Tamb', data=df)
    plt.title('Relative Humidity vs Temperature')
    plt.xlabel('Relative Humidity (%)')
    plt.ylabel('Temperature (°C)')
    
    plt.subplot(1, 2, 2)
    sb.scatterplot(x='RH', y='GHI', data=df, label='GHI', color='orange')
    sb.scatterplot(x='RH', y='DNI', data=df, label='DNI', color='blue')
    sb.scatterplot(x='RH', y='DHI', data=df, label='DHI', color='green')
    plt.title('Relative Humidity vs Solar Radiation')
    plt.xlabel('Relative Humidity (%)')
    plt.ylabel('Solar Radiation (W/m²)')
    plt.legend()
    plt.tight_layout()
    plt.show()
def frequency_distribution(df):
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    sb.histplot(df['GHI'], bins=30, kde=True, color='skyblue')
    plt.title('Histogram of GHI')
    plt.xlabel('GHI (W/m²)')
    plt.ylabel('Frequency')

    plt.subplot(2, 3, 2)
    sb.histplot(df['DNI'], bins=30, kde=True, color='orange')
    plt.title('Histogram of DNI')
    plt.xlabel('DNI (W/m²)')
    plt.ylabel('Frequency')
    
    plt.subplot(2, 3, 3)
    sb.histplot(df['DHI'], bins=30, kde=True, color='green')
    plt.title('Histogram of DHI')
    plt.xlabel('DHI (W/m²)')
    plt.ylabel('Frequency')
    
    plt.subplot(2, 3, 4)
    sb.histplot(df['WS'], bins=30, kde=True, color='purple')
    plt.title('Histogram of Wind Speed')
    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel('Frequency')
    
    plt.subplot(2, 3, 5)
    sb.histplot(df['Tamb'], bins=30, kde=True, color='red')
    plt.title('Histogram of Temperature')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def bubble_chart(df):
    plt.figure(figsize=(12, 8))
    
    # Plot GHI vs. Tamb vs. WS, with bubble size representing RH
    plt.subplot(1, 2, 1)
    plt.scatter(df['Tamb'], df['GHI'], s=df['RH']*10, alpha=0.5, c=df['WS'], cmap='viridis', edgecolor='w', linewidth=0.5)
    plt.colorbar(label='Wind Speed (m/s)')
    plt.title('GHI vs. Temperature vs. Wind Speed (Bubble Size: RH)')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('GHI (W/m²)')

    plt.subplot(1, 2, 2)
    plt.scatter(df['Tamb'], df['GHI'], s=df['BP']*0.1, alpha=0.5, c=df['WS'], cmap='viridis', edgecolor='w', linewidth=0.5)
    plt.colorbar(label='Wind Speed (m/s)')
    plt.title('GHI vs. Temperature vs. Wind Speed (Bubble Size: BP)')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('GHI (W/m²)')
    
    plt.tight_layout()
    plt.show()
