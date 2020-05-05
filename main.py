import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


df = pd.read_csv('/root/Desktop/turkiye-student-evaluation_generic.csv')
print(df.head())
print(df.tail())
print(df.describe())
print(df.isnull())
print(df.corr())


hm = df.corr()
print(sns.heatmap(hm))


print(df['nb.repeat'].value_counts())
print(df.difficulty.value_counts())


plt.figure(figsize=(20, 6))
print(sns.countplot(x = 'class',data = df))


plt.figure(figsize=(20, 10))
print(sns.boxplot(data=df.iloc[:,5:32]))


# CLUSTERRING DATA
X = question_df = df.iloc[:,5:33]
print(question_df.head())

pca = PCA(n_components=2)
PC = pca.fit_transform(question_df)
wcss = []
for i in range(1,8):
    kmeans=KMeans(n_clusters=i,init='k-means++')
    kmeans.fit(PC)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,8),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
print(plt.show())


kmeans = KMeans(n_clusters = 3, init = 'k-means++')
y_kmeans = kmeans.fit_predict(PC)
print(y_kmeans)


plt.scatter(PC[:, 0], PC[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)


sns.kdeplot(PC[:, 0], PC[:, 1], c = y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
sns.kdeplot(centers[:, 0], centers[:, 1], c = 'black', s=200, alpha=0.5)
