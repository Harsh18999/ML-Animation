import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import make_classification
from scipy.spatial import distance
from collections import Counter

def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1-x2)**2))
    return distance

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _predict(self, x):
        # compute the distance
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # get the closest k
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.X_train[i] for i in k_indices]

        return k_nearest_labels

# Function to plot KNN animation
def animate_knn(X, y, new_point, k_max):
    frames = []
    
    # Range for decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # Iterate over k from 1 to k_max to create animation frames
    for k in range(1, k_max + 1):
        # Get the prediction at each step for the new point
        Model = KNN(k)
        Model.fit(X, y)
        
        neighbors = Model._predict(new_point) 
      
        # Create frame showing the current nearest neighbors and decision
        trace_neighbors = go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(color=y,colorscale = ['blue','orange']), name='Data Points') 
        

        trace_nearest = go.Scatter(
            x=[p[0] for p in neighbors],
            y=[p[1] for p in neighbors],
            mode='markers',
            marker=dict(color='black', size=10, symbol='circle-x-open'),
            name=f'{k}-Nearest Neighbors'
        )

        # Show the new point being classified
        trace_new_point = go.Scatter(
            x=[new_point[0]],
            y=[new_point[1]],
            mode='markers',
            marker=dict(color='Red', size=10, symbol='x'),
            name=f'New Point (k={k})'
        )

        frames.append(go.Frame(data=[trace_neighbors, trace_nearest, trace_new_point], layout=go.Layout(title_text=f'K = {k}, Predicted Class: ')))

    # Initial scatter plot before animation starts
    initial_trace_neighbors = go.Scatter(
        x=X[:, 0],
        y=X[:, 1],
        mode='markers',
        marker=dict(color=y, colorscale=['blue', 'orange'], size=10, line=dict(width=0.2)),
        name='Data Points'
    )
    
    initial_trace_new_point = go.Scatter(
        x=[new_point[0]],
        y=[new_point[1]],
        mode='markers',
        marker=dict(color='Red', size=12, symbol='x'),
        name='Test Point'
    )

    fig = go.Figure(
        data=[initial_trace_neighbors, initial_trace_new_point],
        layout=go.Layout(
            title="K-Nearest Neighbors Animation",
            xaxis=dict(range=[x_min, x_max]),
            yaxis=dict(range=[y_min, y_max]),
            updatemenus=[{
                "buttons": [{
                    "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}],
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

# Function to generate data and run KNN animation
def plot_knn_model(n_samples,sep,k,no_classes):
    # Generate synthetic dataset
    X, y = make_classification(n_samples=n_samples, n_features=no_classes, n_classes=no_classes, n_informative=no_classes, n_redundant=0, 
                               n_clusters_per_class=1, hypercube=False, random_state=41, class_sep=sep)
    
    # New point to classify
    new_point = X[np.random.randint(0,X.shape[0])]
    # Animate KNN for k values from 1 to 10
    animate_knn(X, y, new_point, k_max=k)

