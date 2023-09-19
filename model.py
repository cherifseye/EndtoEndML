import numpy as np
from tqdm import tqdm

class LogisticRegression:
    """
    A simple implementation of Logistic Regression using gradient descent.
    
    Parameters:
        learning_rate (float): The learning rate for gradient descent. Default is 0.01.
        num_iterations (int): The number of iterations for gradient descent. Default is 1000.
    """
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.loss = []

    @staticmethod
    def sigmoid(z):
        """
        Sigmoid activation function.

        Parameters:
            z (float or numpy array): The input to the sigmoid function.

        Returns:
            float or numpy array: The output of the sigmoid function.
        """
        if isinstance(z, (int, float)):  # Handle scalar input
            if z >= 0:
                return 1 / (1 + np.exp(-z))
            else:
                return np.exp(z) / (1 + np.exp(z))
        else:  # Handle array input
            sigmoid_values = np.empty_like(z)
            positive_mask = z >= 0
            sigmoid_values[positive_mask] = 1 / (1 + np.exp(-z[positive_mask]))
            sigmoid_values[~positive_mask] = np.exp(z[~positive_mask]) / (1 + np.exp(z[~positive_mask]))
            return sigmoid_values
        f
    def fit(self, X, y):
        """
        Fit the logistic regression model to the given training data.
        
        Parameters:
            X (numpy array): Training data features of shape (num_samples, num_features).
            y (numpy array): Target values of shape (num_samples,).
        """
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in tqdm(range(self.num_iterations), desc="Training", unit="iteration"):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)
            dw = (1/num_samples) * np.dot(X.T, (y_pred - y))
            db = (1/num_samples) * np.sum(y_pred - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            self.loss.append(self.calculate_loss(y, y_pred))

    @staticmethod
    def calculate_loss(y, y_pred):
        epsilon = 1e-15  # Small value to prevent log(0)
        return -np.mean(y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon))

    def predict(self, X):
        """
        Predict the class labels for the given data.
        
        Parameters:
            X (numpy array): Data features of shape (num_samples, num_features).
        
        Returns:
            list: Predicted class labels (0 or 1) for each sample.
        """
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        y_pred_class = [1 if i > 0.5 else 0 for i in y_pred]
        return y_pred_class
    
    def predict_proba(self, X):
        """
        Predict the class probabilities for the given data.

        Parameters:
            X (numpy array): Data features of shape (num_samples, num_features).

        Returns:
            numpy array: Predicted class probabilities for each sample and class.
                         Shape: (num_samples, num_classes)
        """
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        return y_pred
