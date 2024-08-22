import pandas as pd

def summary_statistics(df):
    return df.describe()
def data_quality_check(df):
    missing_values = df.isnull().sum()
    negative_values = (df[['GHI', 'DNI', 'DHI']] < 0).sum()
    return missing_values, negative_values