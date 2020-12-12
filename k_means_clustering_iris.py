################### K-means clustering on Iris dataset  ########################
##################### By Lionel Ngobesing Alangeh ##############################


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.metrics import classification_report

# Importing the datasets
dataset = pd.read_csv('iris2.csv')
X = dataset.iloc[:,[0,1,2,3]].values
dataset['Species'] = pd.Categorical(dataset["Species"])
dataset["Species"] = dataset["Species"].cat.codes

#optimum number of clusters for K-Means classification
from sklearn.cluster  import KMeans
wcss = []
for i in range(1,11):
	kmeans= KMeans(n_clusters= i, init='k-means++', max_iter=400, n_init=10, random_state=0)
	kmeans.fit(X)
	wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Applyig Kmeans to the dataset
kmeans = KMeans(n_clusters= 3, init='k-means++', max_iter=400, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(X)
target_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
cat = classification_report(dataset['Species'],kmeans.labels_,target_names=target_names)
# Visualizing 
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s = 100, c = 'green', label = 'Iris-virginica')

#Cluster Centroid
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1] ,s = 400, c = 'yellow', label = 'Centroid')
plt.title('Cluster of Species')
plt.legend()
plt.show()

#Display kmeans cluster positions and the classification report
print (kmeans.cluster_centers_)
print (cat)
