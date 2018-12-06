#exploratory clustering

import numpy as np
import pandas as pd;
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import StandardScaler  # For scaling dataset
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation #For clustering
from sklearn.mixture import GaussianMixture #For GMM clustering



dataset=pd.read_csv('dummytrain.csv',na_values=['NaN'],sep='\t')


#Scaling
ss = StandardScaler()
ss.fit_transform(dataset)


# K means clustering
def doKmeans(X, nclust=2):
    model = KMeans(nclust)
    model.fit(X)
    clust_labels=model.predict(X)
    cent = model.cluster_centers_
    return (clust_labels, cent)

clust_labels, cent = doKmeans(dataset,5)
kmeans = pd.DataFrame(clust_labels)
dataset.insert((dataset.shape[1]),'kmeans',kmeans)

fig, axes = plt.subplots(2,2,figsize=(12,12))
#Plot the clusters obtained using k means
scatter = axes[0,0].scatter(dataset['GrLivArea'],dataset['SalePrice']/1000,
                     c=kmeans[0],s=50)
axes[0,0].set_title('K-Means Clustering')
axes[0,0].set_xlabel('GrLivArea (sf)')
axes[0,0].set_ylabel('SalePrice (K)')
#plt.savefig('../Figure/4Report/KMeans.png')


#Hierarchical clustering
def doAgglomerative(X, nclust=2):
    model = AgglomerativeClustering(n_clusters=nclust, affinity = 'euclidean', linkage = 'ward')
    clust_labels1 = model.fit_predict(X)
    return (clust_labels1)

clust_labels1 = doAgglomerative(dataset, 5)
agglomerative = pd.DataFrame(clust_labels1)
dataset.insert((dataset.shape[1]),'agglomerative',agglomerative)

#Plot the clusters obtained using Agglomerative clustering or Hierarchical clustering
scatter = axes[0,1].scatter(dataset['GrLivArea'],dataset['SalePrice']/1000,
                     c=agglomerative[0],s=50)
axes[0,1].set_title('Agglomerative Clustering')
axes[0,1].set_xlabel('GrLivArea (sf)')
axes[0,1].set_ylabel('SalePrice (K)')
#plt.savefig('../Figure/4Report/Hierarchical.png')

#affinity propagation
def doAffinity(X):
    model = AffinityPropagation(damping = 0.5, max_iter = 250, affinity = 'euclidean')
    model.fit(X)
    clust_labels2 = model.predict(X)
    cent2 = model.cluster_centers_
    return (clust_labels2, cent2)

clust_labels2, cent2 = doAffinity(dataset)
affinity = pd.DataFrame(clust_labels2)
dataset.insert((dataset.shape[1]),'affinity',affinity)

#Plotting the cluster obtained using Affinity algorithm
scatter = axes[1,0].scatter(dataset['GrLivArea'],dataset['SalePrice']/1000,
                     c=affinity[0],s=50)
axes[1,0].set_title('Affinity Clustering')
axes[1,0].set_xlabel('GrLivArea (sf)')
axes[1,0].set_ylabel('SalePrice (K)')
#plt.savefig('../Figure/4Report/AffinityPropagation.png')


#Gaussian Mixture Modelling
def doGMM(X, nclust=2):
    model = GaussianMixture(n_components=nclust,init_params='kmeans')
    model.fit(X)
    clust_labels3 = model.predict(X)
    return (clust_labels3)

clust_labels3 = doGMM(dataset,5)
gmm = pd.DataFrame(clust_labels3)
dataset.insert((dataset.shape[1]),'gmm',gmm)

#Plotting the cluster obtained using GMM
scatter = axes[1,1].scatter(dataset['GrLivArea'],dataset['SalePrice']/1000,
                     c=gmm[0],s=50)
axes[1,1].set_title('Gaussian Mixture Clustering')
axes[1,1].set_xlabel('GrLivArea (sf)')
axes[1,1].set_ylabel('SalePrice (K)')
#plt.colorbar(scatter)
plt.savefig('../Figure/4Report/Clustering.png')
