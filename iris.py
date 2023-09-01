#Importing Libraries
from model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from tabulate import tabulate

print('                                      ---------------------------------------------------                                          ')
#Getting the dataset
iris = load_iris(as_frame=True)
X, y = iris.data, iris.target
class_mapping = {0: 'iris setosa', 1: 'iris versicolor', 2: 'iris virginica'}
y = y.map(class_mapping)
X["classes"] = y
print('Datasets Visualization')
print(f"Dataset Shape: {X.shape}") 
print("Head of the Data: ")
#Display a portion of the dataset
print(X.head(20))

#Displaying some Information
print("Dataset info")
print(X.info())
#Displaying statistics
print("Statistic")
print(X.describe())

#Display the class Distribution
print(X.groupby('classes').size())

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Plot boxplots for each feature
for i, col in enumerate(X.columns[:-1]):
    ax = axes[i // 2, i % 2]
    X.boxplot(column=col, by='classes', ax=ax)
    ax.set_title(f'Boxplot for {col}')

plt.tight_layout()
plt.show()

X.hist(figsize=(12, 8), bins=50)
plt.show()

# looks for correlation
scatter_matrix(X)
plt.show()

print("Model Training")
iris1 = load_iris()
X, y = iris1.data, iris1.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

y_train_setosa, y_train_versicolor, y_train_virginica = (y_train == 0).astype(np.int8), (y_train == 1).astype(np.int8), (y_train == 2).astype(np.int8)

#Defining model for each classification
setosa_model = LogisticRegression()
versicolor_model = LogisticRegression(num_iterations=40000)
virginica_model = LogisticRegression()
setosa_model.fit(X_train, y_train_setosa)

loss_itosa = setosa_model.loss
y_train_setosa_pred = setosa_model.predict(X_train)
precision_itosa = precision_score(y_train_setosa, y_train_setosa_pred)
recall_itosa = recall_score(y_train_setosa, y_train_setosa_pred)

#Iris versicolor training
versicolor_model.fit(X_train, y_train_versicolor)
loss_versi = versicolor_model.loss
y_train_versicolor_pred = versicolor_model.predict(X_train)
precision_versi = precision_score(y_train_versicolor, y_train_versicolor_pred)
recall_versi = recall_score(y_train_versicolor, y_train_versicolor_pred)

#Iris virginica training
virginica_model.num_iterations = 1000
virginica_model.fit(X_train, y_train_virginica)
loss_virgi = virginica_model.loss
y_train_virginica_pred = virginica_model.predict(X_train)
precision_virgi = precision_score(y_train_virginica ,y_train_virginica_pred)
recall_virgi = recall_score(y_train_virginica, y_train_virginica_pred)


print("Table of information after training")
table_data = [
    ["Kind", "Model", "Loss", "Precision Score", "Recall Score", "iterations"],
    ["Iris Setosa", "Logistic Regression", round(loss_itosa, 4), precision_itosa, recall_itosa, 1000],
    ["Iris Versicolor", "Logistic Regression", round(loss_versi, 4), precision_versi, recall_versi, 40000],
    ["Iris Virginica", "Logistic Regression", round(loss_virgi, 4), precision_virgi, recall_virgi, 1000]
]

# Format and print the table
table = tabulate(table_data, headers="firstrow", tablefmt="fancy_grid")
print(table)
# On test DataSet
y_pred_setosa = setosa_model.predict_proba(X_test)
y_pred_versicolor = versicolor_model.predict_proba(X_test)
y_pred_virginica = virginica_model.predict_proba(X_test)

# Combine predictions to determine the final class labels
y_pred = np.argmax(np.vstack([y_pred_setosa, y_pred_versicolor, y_pred_virginica]).T, axis=1)

# Calculate accuracy on the test dataset
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")