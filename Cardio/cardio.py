import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(script_dir, '..')
sys.path.insert(0, parent_dir)
from process import *
from model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pandas.plotting import scatter_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


cardio_data = read_file('/Users/castekcu/Documents/MachineLearning/Cardio/cardio_train.csv')
showInfo(cardio_data, tail=5)
print("PLotting histograms of Cardio Dataset")
cardio_data.hist(figsize=(20, 10), bins=50)
plt.show()
print("Box Plotting")
box_plotting(cardio_data, by_='cardio')
print("Droping ID column")
cardio_data = cardio_data.drop('id', axis=1)
print(cardio_data.columns)
print("Correlation")
corr_matrix = cardio_data.corr()
print(corr_matrix['cardio'])
cardio_outcome = cardio_data['cardio'].copy()
cardio_train_set = cardio_data.drop(['cardio', 'smoke', 'active'], axis=1)
cardio_train_set, cardio_test_Set, cardio_outcome_train, cardio_outcome_test = train_test_split(cardio_train_set, cardio_outcome, test_size=0.2, random_state=42)

models = {
    'Logistic Regression': LogisticRegression(num_iterations=10000),
    'Random Forest': RandomForestClassifier(),
    'Support Vector Machine': SVC(),
    'K-Nearest Neighbors': KNeighborsClassifier()
}

results_df = pd.DataFrame(columns=['Model', 'Accuracy'])
# Train and evaluate each model
for model_name, model in models.items():
    model.fit(cardio_train_set, cardio_outcome_train)
    y_pred = model.predict(cardio_test_Set)
    accuracy = accuracy_score(cardio_outcome_test, y_pred)
    print(model_name, accuracy)
