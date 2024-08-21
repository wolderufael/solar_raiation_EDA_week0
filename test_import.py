import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from scripts.analysis_script import *

df = pd.read_csv('data/benin-malanville.csv')
print(df.head())
