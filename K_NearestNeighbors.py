import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import make_classification
from scipy.spatial import distance

class KNearestNighbors:

  def __init__(self,distance_matrix,n_neighbors):
    self.distance_matrix = distance_matrix
    self.n_neighbors = n_neighbors

  def get_distance_matrix(self,x_train,test_point):
    dis = 0

    if self.distance_matrix == 'euclidian':
      for i in range(len(x_train)-1):
        dis += (x_train[i] - test_point[i])**2
      euclidian_distance = np.sqrt(dis)
      return euclidian_distance
    
    elif self.distance_matrix == 'manhattan':
      for i in range(len(x_train)-1):
        dis += abs(x_train[i]-test_point[i])
      return dis
    
    else: raise NameError

  def Predict(self,x_train,test_point):

    distance_list = []

    for training_data in x_train:
      distance = self.get_distance_matrix(training_data,test_point)
      distance_list.append([training_data,distance])

    distance_list.sort(key=lambda x:x[1])
    nearest_nighbors = [distance_list[x][0] for x in range(self.n_neighbors)]
    
    return nearest_nighbors

# Function to plot KNN animation
def animate_knn(X, y, new_point, k_max):
    frames = []
    
    # Range for decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # Iterate over k from 1 to k_max to create animation frames
    for k in range(1, k_max + 1):
        # Get the prediction at each step for the new point
        neighbors = KNearestNighbors('manhattan',k).Predict(X,new_point)
  
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
    X, y = make_classification(n_samples=n_samples, n_features=2, n_classes=2, n_informative=2, n_redundant=0, 
                               n_clusters_per_class=1, hypercube=False, random_state=41, class_sep=sep)
    
    # New point to classify
    new_point = X[np.random.randint(0,X.shape[0])]
    # Animate KNN for k values from 1 to 10
    animate_knn(X, y, new_point, k_max=k)

