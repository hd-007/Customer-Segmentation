# -*- coding: utf-8 -*-
"""
Created on Thu May  7 18:28:59 2020

@author: UKL
"""


import pandas as pd
import numpy as np

import visuals5 as vs
import seaborn as sns

data=pd.read_csv('data.csv', sep='\t')
#print(data.head())
data=data.drop(['Region', 'Channel'], axis = 1)
print("Wholesale customers dataset has {} samples with {} features each.".format(*data.shape))

display(data.describe())

indices = [1,50,100]

# Create a DataFrame of the chosen samples
samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)
print("Chosen samples of wholesale customers dataset:")
display(samples)

print ("Offset from mean of whole dataset")
display(samples - np.around(data.mean()))

print ("Offset from median of whole dataset:")
display(samples - np.around(data.median()))

print("Samples vs Dataset Mean Values")
#sns.heatmap((samples-np.around(data.mean().values))/data.std(ddof=0), annot=True)

#print(((samples-np.around(data.mean().values))/data.std()))

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
# TODO: Make a copy of the DataFrame, using the 'drop' function to drop the given feature

for col in data.columns:
    
    new_data = data.drop(col, axis=1, inplace = False)

    # TODO: Split the data into training and testing sets(0.25) using the given feature as the target
    # Set a random state.
    X_train, X_test, y_train, y_test =  train_test_split(new_data, data[col], test_size=0.25, random_state=42)

# TODO: Create a decision tree regressor and fit it to the training set
    regressor = DecisionTreeRegressor(random_state=42)
    regressor.fit(X_train,y_train)
    
# TODO: Report the score of the prediction using the testing set
    score = regressor.score(X_test,y_test)
    print (col, score)
    
pd.plotting.scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde')
a=data.corr()
print(a)

sns.heatmap(data.corr(), annot=True)

log_data = np.log(data)

# TODO: Scale the sample data using the natural logarithm
log_samples = np.log(samples)

# Produce a scatter matrix for each pair of newly-transformed features
pd.plotting.scatter_matrix(log_data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');

display(log_samples)

#sns.heatmap(log_data.corr())

for feature in log_data.keys():
    
    print("fea",feature)
    # TODO: Calculate Q1 (25th percentile of the data) for the given feature
    Q1 = np.percentile(log_data[feature],25)
    print("1",Q1)
    # TODO: Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = np.percentile(log_data[feature],75)
    print("3",Q3)
    
    step = (Q3-Q1)*1.5
    print("step",step)
    # Display the outliers
    print("Data points considered outliers for the feature '{}':".format(feature))
    display(log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))])

outliers  = [65,66,75,128,154]

# Remove the outliers, if any were specified
good_data = log_data.drop(log_data.index[outliers]).reset_index(drop = True)

print(log_data.shape, good_data.shape)

# TODO: Apply PCA by fitting the good data with the same number of dimensions as features
from sklearn.decomposition import PCA
pca = PCA(n_components=len(good_data.columns)).fit(good_data)

# TODO: Transform log_samples using the PCA fit above
pca_samples = pca.transform(log_samples)

# Generate PCA results plot
pca_results = vs.pca_results(good_data, pca)

display(pca_results)

# DataFrame
display(type(pca_results))

# Cumulative explained variance should add to 1
display(pca_results['Explained Variance'].cumsum())

display(pd.DataFrame(np.round(pca_samples, 4), columns = pca_results.index.values))

# TODO: Apply PCA by fitting the good data with only two dimensions
pca = PCA(n_components=2).fit(good_data)

# TODO: Transform the good data using the PCA fit above
reduced_data = pca.transform(good_data)

# TODO: Transform log_samples using the PCA fit above
pca_samples = pca.transform(log_samples)

# Create a DataFrame for the reduced data
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])

display(pd.DataFrame(np.round(pca_samples, 4), columns = ['Dimension 1', 'Dimension 2']))

#vs.biplot(good_data, reduced_data, pca)

# TODO: Apply your clustering algorithm of choice to the reduced data 
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


range_n_clusters = list(range(2,11))

def getScores(num_clusters):
    clusterer = GMM(n_components=num_clusters, random_state=42).fit(reduced_data)

    # TODO: Predict the cluster for each data point
    preds = clusterer.predict(reduced_data)

    # TODO: Find the cluster centers
    centers = clusterer.means_

    # TODO: Predict the cluster for each transformed sample data point
    sample_preds = clusterer.predict(pca_samples)

    # TODO: Calculate the mean silhouette coefficient for the number of clusters chosen
    score = silhouette_score(reduced_data,preds)
    return score

lowest_bic = np.infty
bic = []
n_components_range = range(1, 7)
cv_types = ['spherical', 'tied', 'full', 'diag' ]
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a mixture of Gaussians with EM
        gmm = GMM(n_components=n_components, covariance_type=cv_type)
        gmm.fit(good_data)
        bic.append(gmm.bic(good_data))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm
            
print(best_gmm)            
'''
for n_clusters in range_n_clusters:
    # TODO: Apply your clustering algorithm of choice to the reduced data 
    clusterer = KMeans(n_clusters=n_clusters).fit(reduced_data)

    # TODO: Predict the cluster for each data point
    preds = clusterer.predict(reduced_data)

    # TODO: Find the cluster centers
    centers = clusterer.cluster_centers_

    # TODO: Predict the cluster for each transformed sample data point
    sample_preds = clusterer.predict(pca_samples)

    # TODO: Calculate the mean silhouette coefficient for the number of clusters chosen
    score = silhouette_score(reduced_data, preds, metric='euclidean')
    print ("For n_clusters = {}. The average silhouette_score is : {}".format(n_clusters, score))
'''
a=[]
scores = pd.DataFrame(columns=['Silhouette Score'])
scores.columns.name = 'Number of Clusters'    
for i in range(2,50):
    score = getScores(i) 
    scores = scores.append(pd.DataFrame([score],columns=['Silhouette Score'],index=[i]))
    a.append(i)
display(scores)

plt.figure(figsize=(9, 3))
plt.plot(a,scores['Silhouette Score'])
plt.show()

clusterer = GMM(n_components=2,covariance_type='full').fit(reduced_data)

# TODO: Predict the cluster for each data point
preds = clusterer.predict(reduced_data)

# TODO: Find the cluster centers
centers = clusterer.means_

# TODO: Predict the cluster for each transformed sample data point
sample_preds = clusterer.predict(pca_samples)

# TODO: Calculate the mean silhouette coefficient for the number of clusters chosen
score = silhouette_score(reduced_data,preds)

print (score)





