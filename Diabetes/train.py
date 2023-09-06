import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(script_dir, '..')
sys.path.insert(0, parent_dir)
import pandas as pd
import numpy as np
from model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def read_file(filename):
    return pd.read_csv(filename)

diabetes = read_file('Diabetes/diabetes.csv')


# Split the data into train and test sets
train_set, test_set = train_test_split(diabetes, test_size=0.2, random_state=42)

# Define the features based on correlation analysis
selected_features = ['Glucose', 'BMI', 'Age']  # Replace with your relevant features

# Prepare the training data
X_train = train_set[selected_features]
y_train = train_set['Outcome']

X_test = test_set[selected_features]
y_test = test_set['Outcome']
models = {
    'Logistic Regression': LogisticRegression(num_iterations=10000),
    'Random Forest': RandomForestClassifier(),
    'Support Vector Machine': SVC(),
    'K-Nearest Neighbors': KNeighborsClassifier()
}

results_df = pd.DataFrame(columns=['Model', 'Accuracy'])
# Train and evaluate each model
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results_df = results_df.append({'Model': model_name, 'Accuracy': accuracy}, ignore_index=True)

results_df.to_csv("Diabetes/Diabetes_models_results.csv", index=False)