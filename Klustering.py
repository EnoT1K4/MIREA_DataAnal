import collections
import random
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from mpl_toolkits.mplot3d import Axes3D

data_1 = pd.read_csv('customer_classification_data.csv')
answers = data_1['Education'].value_counts()
data = data_1.drop('Education', axis=1)
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data)

def kmeans_all(normalized_data):
    inertia_values = []
    silhouette_scores = []
    k_val = range(2,11)
    for k in k_val:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(normalized_data)
        inertia_values.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(normalized_data, kmeans.labels_))
        
    plt.plot(k_val, inertia_values, 'bo-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Метод локтя')
    plt.show()

    plt.plot(k_val, silhouette_scores, 'bo-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('коэфф силуэта')
    plt.show()

    best_k = 4
    best_kmeans = KMeans(n_clusters=best_k)
    best_kmeans.fit(normalized_data)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(normalized_data[:, 0], normalized_data[:, 1], normalized_data[:, 2], c=kmeans.labels_)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    ax.set_title('Cluster Visualization')
    plt.show()

    cluster_counts = np.bincount(best_kmeans.labels_)
    for cluster, count in enumerate(cluster_counts):
        print(f'Количество элементов в кластере {cluster}: {count}, в первичном датасете: {answers[cluster]}')
        
    return ('--------------------------------')
def dbscan_all(normalized_data):
    
    dbscan = DBSCAN(eps=1.0, min_samples=50)
    clusters = dbscan.fit_predict(normalized_data)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(normalized_data[:, 0], normalized_data[:, 1], normalized_data[:, 2], c=clusters, cmap='rainbow')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    ax.set_title('DBSCAN Clustering Visualization')
    plt.show()

    cluster_counts = np.bincount(clusters+1)
    for cluster, count in enumerate(cluster_counts):
        print(f'Количество элементов в кластере {cluster}: {count}, в первичном датасете: {answers[cluster]}')   
    return('--------------------------------')
def hierarchy_all(normalized_data):
    Z = linkage(normalized_data, method='complete', metric='euclidean')
    plt.figure(figsize=(10, 5))
    dendrogram(Z)
    plt.xlabel('Objects')
    plt.ylabel('Distance')
    plt.title('Hierarchical Clustering Dendrogram')
    plt.show()

    threshold = 3
    labels = fcluster(Z, threshold, criterion='maxclust')
    cluster_counts = np.bincount(labels)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(normalized_data[:, 0], normalized_data[:, 1], normalized_data[:, 2], c=labels, cmap='rainbow')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    ax.set_title('Hierarchy Clustering Visualization')
    plt.show()

    for cluster, count in enumerate(cluster_counts):
        print(f'Количество элементов в кластере {cluster}: {count}, в первичном датасете: {answers[cluster]}')
    return ('------------------------------------------------')
        
print(kmeans_all(normalized_data))
print(dbscan_all(normalized_data))
print(hierarchy_all(normalized_data))
print(answers)