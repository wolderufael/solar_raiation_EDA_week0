import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

file1 = "../data/benin-malanville.csv"
file2 = "../data/sierraleone-bumbuna.csv"
file3 = "../data/togo-dapaong_qc.csv"

df_benin = pd.read_csv(file1)
df_sierraleone = pd.read_csv(file2)
df_togo = pd.read_csv(file3)

# Dictionary to map file names to actual file paths
file_dict = {
    "Benin": file1,
    "Sierraleone": file2,
    "Togo": file3
}

def plot_daily_mean(df1, df2, df3):
    # Resample each DataFrame to daily means
    df1.index = pd.to_datetime(df1['Timestamp'])
    df2.index = pd.to_datetime(df2['Timestamp'])
    df3.index = pd.to_datetime(df3['Timestamp'])
    
    df1 = df1.drop(columns=['Timestamp'])
    df2 = df2.drop(columns=['Timestamp'])
    df3 = df3.drop(columns=['Timestamp'])

    daily_mean_df1 = df1.resample('D').mean()
    daily_mean_df2 = df2.resample('D').mean()
    daily_mean_df3 = df3.resample('D').mean()
    
    variables = ['GHI', 'DNI', 'DHI']
    
    for var in variables:
        plt.figure(figsize=(10, 6))
        plt.plot(daily_mean_df1.index, daily_mean_df1[var], label='Benin')
        plt.plot(daily_mean_df2.index, daily_mean_df2[var], label='Sierraleone')
        plt.plot(daily_mean_df3.index, daily_mean_df3[var], label='Togo')
        plt.title(f'Daily Mean {var}')
        plt.xlabel('Date')
        plt.ylabel(var)
        plt.legend()
        st.pyplot(plt.gcf())  # This integrates the plot within the Streamlit app
def mean_median_std_comparison(df1, df2, df3):
    # Mean comparison plot of GHI
    mean_ghi_1 = df1['GHI'].mean()
    mean_ghi_2 = df2['GHI'].mean()
    mean_ghi_3 = df3['GHI'].mean()
    
    data = {
        'Site': ['Benin', 'Sierraleone', 'Togo'],
        'Mean GHI': [mean_ghi_1, mean_ghi_2, mean_ghi_3]
    }
    mean_df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    bars = plt.bar(mean_df['Site'], mean_df['Mean GHI'], color=['blue', 'green', 'orange'])
    plt.bar_label(bars, fmt='%.2f')
    
    plt.title('Comparison of Mean GHI Across Three Sites')
    plt.xlabel('Site')
    plt.ylabel('Mean GHI')
    st.pyplot(plt.gcf())  # Integrate the plot within Streamlit

    # Standard deviation comparison plot of GHI
    std_ghi_1 = df1['GHI'].std()
    std_ghi_2 = df2['GHI'].std()
    std_ghi_3 = df3['GHI'].std()
    
    data = {
        'Site': ['Benin', 'Sierraleone', 'Togo'],
        'Std GHI': [std_ghi_1, std_ghi_2, std_ghi_3]
    }
    std_df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    bars = plt.bar(std_df['Site'], std_df['Std GHI'], color=['blue', 'green', 'orange'])
    plt.bar_label(bars, fmt='%.2f')
    
    plt.title('Comparison of Standard Deviation of GHI Across Three Sites')
    plt.xlabel('Site')
    plt.ylabel('Std GHI')
    st.pyplot(plt.gcf())  # Integrate the plot within Streamlit

    ###########################################
    # Mean comparison plot of DNI
    mean_dni_1 = df1['DNI'].mean()
    mean_dni_2 = df2['DNI'].mean()
    mean_dni_3 = df3['DNI'].mean()
    
    data = {
        'Site': ['Benin', 'Sierraleone', 'Togo'],
        'Mean DNI': [mean_dni_1, mean_dni_2, mean_dni_3]
    }
    mean_df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    bars = plt.bar(mean_df['Site'], mean_df['Mean DNI'], color=['blue', 'green', 'orange'])
    plt.bar_label(bars, fmt='%.2f')
    
    plt.title('Comparison of Mean DNI Across Three Sites')
    plt.xlabel('Site')
    plt.ylabel('Mean DNI')
    st.pyplot(plt.gcf())  # Integrate the plot within Streamlit

    # Standard deviation comparison plot of DNI
    std_dni_1 = df1['DNI'].std()
    std_dni_2 = df2['DNI'].std()
    std_dni_3 = df3['DNI'].std()
    
    data = {
        'Site': ['Benin', 'Sierraleone', 'Togo'],
        'Std DNI': [std_dni_1, std_dni_2, std_dni_3]
    }
    std_df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    bars = plt.bar(std_df['Site'], std_df['Std DNI'], color=['blue', 'green', 'orange'])
    plt.bar_label(bars, fmt='%.2f')
    
    plt.title('Comparison of Standard Deviation of DNI Across Three Sites')
    plt.xlabel('Site')
    plt.ylabel('Std DNI')
    st.pyplot(plt.gcf())  # Integrate the plot within Streamlit

    ###########################################
    # Mean comparison plot of DHI
    mean_dhi_1 = df1['DHI'].mean()
    mean_dhi_2 = df2['DHI'].mean()
    mean_dhi_3 = df3['DHI'].mean()
    
    data = {
        'Site': ['Benin', 'Sierraleone', 'Togo'],
        'Mean DHI': [mean_dhi_1, mean_dhi_2, mean_dhi_3]
    }
    mean_df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    bars = plt.bar(mean_df['Site'], mean_df['Mean DHI'], color=['blue', 'green', 'orange'])
    plt.bar_label(bars, fmt='%.2f')
    
    plt.title('Comparison of Mean DHI Across Three Sites')
    plt.xlabel('Site')
    plt.ylabel('Mean DHI')
    st.pyplot(plt.gcf())  # Integrate the plot within Streamlit

    # Standard deviation comparison plot of DHI
    std_dhi_1 = df1['DHI'].std()
    std_dhi_2 = df2['DHI'].std()
    std_dhi_3 = df3['DHI'].std()
    
    data = {
        'Site': ['Benin', 'Sierraleone', 'Togo'],
        'Std DHI': [std_dhi_1, std_dhi_2, std_dhi_3]
    }
    std_df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    bars = plt.bar(std_df['Site'], std_df['Std DHI'], color=['blue', 'green', 'orange'])
    plt.bar_label(bars, fmt='%.2f')
    
    plt.title('Comparison of Standard Deviation of DHI Across Three Sites')
    plt.xlabel('Site')
    plt.ylabel('Std DHI')
    st.pyplot(plt.gcf())
# Streamlit app
st.title("CSV Data Viewer")

# Sidebar with options
st.sidebar.title("Analysis Options")
selected_analysis = st.sidebar.radio("Select Analysis", ["Summary Statistics", "Comparison", "Time Series Analysis"])

# Dropdown to select the CSV file
selected_file = st.selectbox("Select a CSV file", list(file_dict.keys()))

# Load the selected CSV file
if selected_file:
    df = pd.read_csv(file_dict[selected_file])

    # Display the first 5 rows of the selected CSV file
    # st.write(df.head())

    # Summary Statistics
    if selected_analysis == "Summary Statistics":
        st.sidebar.subheader("Summary Statistics")
        st.write(df.describe())

    # Comparison Analysis
    elif selected_analysis == "Comparison":
        st.sidebar.subheader("Comparison Analysis")
        st.write(mean_median_std_comparison(df_benin, df_sierraleone, df_togo))
       
    # Time Series Analysis
    elif selected_analysis == "Time Series Analysis":
        st.sidebar.subheader("Time Series Analysis")
        plot_daily_mean(df_benin, df_sierraleone, df_togo)
    
