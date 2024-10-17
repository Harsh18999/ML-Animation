import plotly.express as px
import numpy as np
from sklearn import datasets
import plotly.graph_objects as go
import streamlit as st


class Linear_Regression():

  # initiating the parameters (learning rate & no. of iterations)
  def __init__(self, learning_rate, no_of_iterations,noise=15,samples=100):

    self.learning_rate = learning_rate
    self.no_of_iterations = no_of_iterations
    X,y = datasets.make_regression(n_samples=samples, n_features=1, n_informative=1, n_targets=1, bias=0.0, effective_rank=None, tail_strength=0.5, noise=noise, shuffle=True, coef=False, random_state=42)
    self.fit(X,y)

  def fit(self, X, Y):
        # number of training examples & number of features
        self.m, self.n = X.shape  # number of rows & columns

        # initiating the weight and bias
        self.w = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        # Create a list to store frames
        frames = []

        # Implementing Gradient Descent
        for i in range(self.no_of_iterations):
            self.update_weights()

            # Create a frame with the current state of the line
            # Assuming X is a single feature, i.e., X.shape = (m, 1)
            y_pred = self.w[0] * X[:, 0] + self.b

            # Append frame (with both scatter plot and the evolving line)
            frames.append(go.Frame(data=[
                go.Scatter(x=X[:, 0], y=Y, mode='markers', name='Data Points'),  # Static scatter plot
                go.Scatter(x=X[:, 0], y=y_pred, mode='lines', name=f'Iteration {i+1}')  # Evolving line
            ],
            layout=go.Layout(title_text=f"Iteration {i+1}")))

        # Define the initial line (before animation starts)
        y_initial = self.w[0] * X[:, 0] + self.b
        Plot(frames,X,Y,y_initial)

  def update_weights(self):

    Y_prediction = self.predict(self.X)

    # calculate gradients

    dw = - (2 * (self.X.T).dot(self.Y - Y_prediction)) / self.m

    db = - 2 * np.sum(self.Y - Y_prediction)/self.m

    # upadating the weights

    self.w = self.w - self.learning_rate*dw
    self.b = self.b - self.learning_rate*db


  def predict(self, X):

    return X.dot(self.w) + self.b

def Plot(frames,X,Y,y_initial):
   fig = go.Figure(
            data=[
                go.Scatter(x=X[:, 0], y=Y, mode='markers', name='Data Points'),  # Static scatter plot
                go.Scatter(x=X[:, 0], y=y_initial, mode='lines', name='Initial Line')  # Initial regression line
            ],
            layout=go.Layout(
                title="Linear Regression Animation",
                updatemenus=[{
                    "buttons": [{
                        "args": [None, {"frame": {"duration": 100, "redraw": True},
                                        "fromcurrent": True}],
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
   st.plotly_chart(fig)
   
