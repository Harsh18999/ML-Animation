import pandas as pd
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from sklearn.datasets import make_classification

class LogisticRegression:
  
    def __init__(self, learning_rate, iterations):
        # Initialize the learning rate and number of iterations
        self.lr = learning_rate
        self.iter = iterations
  
    def fit(self, X, y):
        # Append a bias term (a column of 1's) to X
        X = np.insert(X, 0, 1, axis=1)

        # Initialize weights (1D array)
        self.w = np.ones(X.shape[1])
    
        # Initialize frames to store the frames of plots
        frames = []
        
        # Set limits for x and y axes
        x_min, x_max = min(X[:, 1]), max(X[:, 1])
        y_min, y_max = min(X[:, 2]), max(X[:, 2])

        # Iterate and update weights using stochastic gradient descent
        for i in range(self.iter):
            # Select a random point to update the weights
            j = np.random.randint(0, X.shape[0])

            # Calculate the sigmoid function for that point
            y_pred = self.sigmoid(np.dot(self.w, X[j]))

            # Update the weights
            self.w += (y[j] - y_pred) * self.lr * X[j]

            # Calculate the decision boundary (y_line) for the current weights
            x_values = np.linspace(x_min, x_max, 100)
            y_line = -(self.w[1] / self.w[2]) * x_values - (self.w[0] / self.w[2])  # Decision boundary formula

            # Append frames for animation
            frames.append(go.Frame(data=[
                go.Scatter(x=X[:, 1], y=X[:, 2], mode='markers', marker=dict(color=y,colorscale = ['blue','orange']), name='Data Points'),  # Scatter plot of the data points
                go.Scatter(x=x_values, y=y_line, mode='lines', name=f'Iteration {i+1}')  # Evolving decision boundary
            ], layout=go.Layout(title_text=f"Iteration {i+1}")))

        # Initial predicted decision boundary before training
        x_initial = np.linspace(x_min, x_max, 100)
        y_initial = -(self.w[1] / self.w[2]) * x_initial - (self.w[0] / self.w[2])
        
        # Call the Plot function to display the animation
        Plot(frames, X, y, x_initial, y_initial, x_min, x_max, y_min, y_max)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))  # Correct sigmoid implementation

def Plot(frames, X, Y, x_initial, y_initial, x_min, x_max, y_min, y_max):
    # Create a Plotly figure for animation with axis limits
    fig = go.Figure(
        data=[
            go.Scatter(x=X[:, 1], y=X[:, 2],marker=dict(color=Y,colorscale = ['blue','orange']), mode='markers', name='Data Points'),  # Static scatter plot
            go.Scatter(x=x_initial, y=y_initial, mode='lines', name='Initial Line')  # Initial decision boundary
        ],
        layout=go.Layout(
            title="Logistic Regression Animation",
            xaxis_range=[x_min, x_max],  # Set x-axis range
            yaxis_range=[y_min, y_max],  # Set y-axis range
            updatemenus=[{
                "buttons": [{
                    "args": [None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}],
                    "label": "Play",
                    "method": "animate"
                }],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }]
        ),
        frames=frames
    )
    st.plotly_chart(fig)  # Display in Streamlit

def Plot_Model(n_samples,sep,lr,iter):
    # Generate a synthetic dataset using sklearn
    X, y = make_classification(n_samples=n_samples, n_features=2, n_classes=2, n_informative=1, n_redundant=0, 
                               n_clusters_per_class=1, hypercube=False, random_state=41, class_sep=sep)
    # Initialize the LogisticRegression model and fit the data
    LogisticRegression(learning_rate=lr, iterations=iter).fit(X, y)
