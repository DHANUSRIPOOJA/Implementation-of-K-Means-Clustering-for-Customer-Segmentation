# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import necessary libraries such as pandas, matplotlib, and sklearn.
2.Load the dataset Mall_Customers.csv.
3.Select the required features for clustering (e.g., Annual Income and Spending Score).
4.Use the Elbow Method to determine the optimal number of clusters (k).
4.Apply K-Means clustering algorithm to group the customers into k clusters.
5.Visualize the clusters using a scatter plot.
6.Analyze the clusters and interpret the results.

## Program:
```

Program to implement the K Means Clustering for Customer Segmentation.
Developed by: K DHANUSRI POOJA
RegisterNumber:  212224040068

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
data = pd.read_csv("/content/Mall_Customers.csv")
data
X = data[['Annual Income (k$)' , 'Spending Score (1-100)']]
X
plt.figure(figsize=(4,4))
plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel("Spending Score (1-100)")
plt.show()
k = 5
kmeans = KMeans(n_clusters=k)
kmeans.fit(X)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_
print("Centroids: ")
print(centroids)
print("Label:")
# define colors for each cluster
colors = ['r', 'g', 'b', 'c', 'm']

# plotting the controls
for i in range(k):
  cluster_points = X[labels == i]
  plt.scatter(cluster_points['Annual Income (k$)'], cluster_points['Spending Score (1-100)'], color=colors[i], label=f'Cluster {i+1}')

  #Find minimum enclosing circle
  distances = euclidean_distances(cluster_points, [centroids[i]])
  radius = np.max(distances)

  circle = plt.Circle(centroids[i], radius, color=colors[i], fill=False)
  plt.gca().add_patch(circle)

#Plotting the centroids
plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', s=200, color='k', label='Centroids')

plt.title('K-means Clustering')
plt.xlabel("Annual Income (k$)")
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.axis('equal') 
plt.show()

```

## Output:

<img width="493" height="429" alt="325288384-de1e047c-8fc0-4e7e-88e3-250bd28ba8c6" src="https://github.com/user-attachments/assets/3c22f5af-09b0-4f06-9b92-7e179f60a85a" />

<img width="670" height="265" alt="325288457-84fd0614-b594-4265-8dec-243ec7f7acc8" src="https://github.com/user-attachments/assets/c93d3329-fe1e-4863-affb-571a4e1e9d7a" />

<img width="721" height="526" alt="325288659-a1c6c16d-8bc7-4cd8-bc3d-cebde586281a" src="https://github.com/user-attachments/assets/95851ebf-a72a-47ba-bd1c-2bf648157d91" />


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
