import streamlit as st
from LinearRegression import Linear_Regression
from LogisticRegression import Plot_Model
from K_NearestNeighbors import plot_knn_model
from K_MeansClustering import plot_kmeans_model

st.title(':red[ML Algorithm] :blue[Animations]')
st.sidebar.title(':red[ML Algorithm] :blue[Animations]')

st.write('''Welcome to :red[ML Algorithm] :blue[Animations]! ''')

st.write("Discover machine learning like never before. This platform brings complex algorithms to life with interactive, easy-to-understand animations. Whether you are a student, data enthusiast, or a seasoned professional, you can explore and visualize the inner workings of various machine learning models, from simple linear regression to advanced neural networks.")
         
st.write(":black[About the Project:]") 
st.write("The ML Animations project is designed to help users grasp the core concepts of machine learning through dynamic, visual representations. Each animation breaks down the steps of the algorithms, offering an intuitive learning experience. Whether you are looking to reinforce your understanding or gain a fresh perspective, this platform provides an engaging way to interact with machine learning models.")


Ml = st.sidebar.selectbox('Select ML type:', options=['Supervised', 'Un-Supervised'])
algorithms = []

if Ml == 'Supervised':
    algorithms = ['Linear Regression', 'Logistic Regression', 'K-Nearest Neighbors']
elif Ml == 'Un-Supervised':
    algorithms = ['K-Means Clustering']

Model = st.sidebar.selectbox('Select Machine Learning Algorithm:', options=algorithms)

if Model == 'Linear Regression':
    samples = st.sidebar.number_input('Enter Number of sample data points', value=100)
    noise = st.sidebar.number_input('Noise in data', value=15)
    lr = st.sidebar.number_input('Enter Learning rate', value=0.1)
    max_iter = st.sidebar.number_input('Enter number of iterations', value=100, step=1)

    if st.sidebar.button('Show'):
        Linear_Regression(learning_rate=lr, noise=noise, no_of_iterations=max_iter, samples=samples)


elif Model == 'Logistic Regression':
    samples = st.sidebar.number_input('Enter Number of sample data points', value=100)
    sep = st.sidebar.number_input('Separation between classes', value=5)
    lr = st.sidebar.number_input('Enter Learning rate', value=0.1)
    max_iter = st.sidebar.number_input('Enter number of iterations', value=100, step=1)

    if st.sidebar.button('Show'):
        Plot_Model(n_samples=samples, sep=sep, lr=lr, iter=max_iter)

elif Model == 'K-Nearest Neighbors':
    samples = st.sidebar.number_input('Enter Number of sample data points', value=100)
    no_classes = st.sidebar.number_input('Enter no of classses',value=2,step=1)
    sep = st.sidebar.number_input('Separation between classes', value=5)
    k = st.sidebar.number_input('Enter value', value=5, step=1)
    max_iter = st.sidebar.number_input('Enter number of iterations', value=100, step=1)

    if st.sidebar.button('Show'):
        plot_knn_model(n_samples=samples, sep=sep,k=k,no_classes=no_classes)

elif Model =='K-Means Clustering':
    samples = st.sidebar.number_input('Enter Number of sample data points', value=100)
    n_clusters = st.sidebar.number_input('Enter no of clusters',value=2,step=1)
    cluster_std = st.sidebar.number_input('Enter cluster std',value=2)
    max_iter = st.sidebar.number_input('Enter number of iterations', value=100, step=1)

    if st.sidebar.button('Show Animation'):
        plot_kmeans_model(n_samples=samples,n_cluster=n_clusters,iter=max_iter,cluster_std=cluster_std)


