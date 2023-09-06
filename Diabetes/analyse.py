import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(script_dir, '..')
sys.path.insert(0, parent_dir)
from process import box_plotting
def read_file(filename):
    return pd.read_csv(filename)

filename = 'Diabetes/diabetes.csv'
print("Loading Data for reading...")
diabetes = read_file(filename)
print("Done")
print("Displaying head of %s", filename)
print(diabetes.head(20))
print("")
print("Displaying Information of %s", filename)
print(diabetes.info())
print("")
print("Description of %s", filename)
print(diabetes.describe())
print("PLotting Hsitogram of %s", filename)
diabetes.hist(figsize=(12, 8), bins=50)
plt.show()
print("Histogram plot Closed")

print("Show Box plotting with the outcome: ")
box_plotting(diabetes, 'Outcome')
print("Box Plots Closed")
#Creating a train and test set
train_set, test_set = train_test_split(diabetes, test_size=.2, random_state=42)
diabetes = train_set.copy()
corr_matrix = diabetes.corr()
print(corr_matrix ['Outcome'].sort_values(ascending=True))
print("Plotting scatter matrix")
scatter_matrix(diabetes)
plt.show()