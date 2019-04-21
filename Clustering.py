# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 14:23:41 2019

@author: kdandebo
"""

import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 500)
df = pd.ExcelFile('C:/Users/Kavya/Desktop/Python/datasetforpractice/EastWestAirlines.xlsx')
df = df.parse("data")
print(df.columns)
print(df.head(10))

#Dropping the column 'ID' before Normalizing the data as it is of no importance
df.drop(['ID#'],axis=1,inplace=True)
print(df.columns)

#Normalizing the data to one scale
from sklearn import preprocessing
standardized_df = preprocessing.scale(df)
#print(standardized_df)
standardized = pd.DataFrame(standardized_df)
df = standardized
df_cols = ['Balance', 'Qual_miles', 'cc1_miles', 'cc2_miles', 'cc3_miles','Bonus_miles', 'Bonus_trans', 'Flight_miles_12mo', 'Flight_trans_12','Days_since_enroll', 'Award?']
#df = df[['Balance', 'Qual_miles', 'cc1_miles', 'cc2_miles', 'cc3_miles','Bonus_miles', 'Bonus_trans', 'Flight_miles_12mo', 'Flight_trans_12','Days_since_enroll', 'Award?']]
df.columns = df_cols
print(df.head(10))
print(df.info())

#chekcing the distance matrix between all the data points
from scipy.spatial import distance_matrix
print(distance_matrix(df.values,df.values))

#or by this method

from scipy.cluster.hierarchy import linkage
z = linkage(df,method = 'complete', metric = 'euclidean')
print(z)

#creating a dendogram

import scipy.cluster.hierarchy as shc
dend = shc.dendrogram(shc.linkage(df, method='complete')) 
plt.title('Hierarchical clustering dhedogram')
plt.xlabel('Airline index number')
plt.ylabel('Dist')
plt.show()

#creating a model for non-hearchial clsutering

from sklearn.cluster import AgglomerativeClustering
Hclustering = AgglomerativeClustering(n_clusters=44, affinity='euclidean', linkage='complete').fit(df)
print(Hclustering)

#we can see how the indexes are distributed to the respective clusters
from collections import Counter
print(Counter(Hclustering.labels_))
ClusterNumbers = pd.DataFrame(Hclustering.labels_)
df['ClusterNumbers'] = ClusterNumbers
df = df[['ClusterNumbers', 'Balance', 'Qual_miles', 'cc1_miles', 'cc2_miles', 'cc3_miles','Bonus_miles', 'Bonus_trans', 'Flight_miles_12mo', 'Flight_trans_12','Days_since_enroll', 'Award?']]
#df.insert(1, "Univ", df['Univ']) 
print(df.head(10))

#adding the output to a csv file
df.to_csv('C:/Users/kdandebo/Desktop/Models/DS training/Yogesh data sets/Airline.csv')

##Inferences of non-heirarchial clustering is 

#1. cluster number 5 has most number of elements. - 2824



#################### K-means clustering ########################


import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 500)
df = pd.ExcelFile('C:/Users/kdandebo/Desktop/Models/DS training/Yogesh data sets/EastWestAirlines.xlsx')
df = df.parse("data")
print(df.columns)
print(df.head(10))

#Dropping the column 'ID' before Normalizing the data as it is of no importance
df.drop(['ID#'],axis=1,inplace=True)
print(df.columns)

#Normalizing the data to one scale
from sklearn import preprocessing
standardized_df = preprocessing.scale(df)
#print(standardized_df)
standardized = pd.DataFrame(standardized_df)
df = standardized
df_cols = ['Balance', 'Qual_miles', 'cc1_miles', 'cc2_miles', 'cc3_miles','Bonus_miles', 'Bonus_trans', 'Flight_miles_12mo', 'Flight_trans_12','Days_since_enroll', 'Award?']
#df = df[['Balance', 'Qual_miles', 'cc1_miles', 'cc2_miles', 'cc3_miles','Bonus_miles', 'Bonus_trans', 'Flight_miles_12mo', 'Flight_trans_12','Days_since_enroll', 'Award?']]
df.columns = df_cols
print(df.head(10))
print(df.info())

#chekcing the distance matrix between all the data points
from scipy.spatial import distance_matrix
print(distance_matrix(df.values,df.values))

#or by this method

from scipy.cluster.hierarchy import linkage
z = linkage(df,method = 'centroid', metric = 'euclidean')
print(z)


#creating a model for K-means clsutering

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=44, random_state=0).fit(df)
print(kmeans.labels_)
print(kmeans)
#we can see how the indexes are distributed to the respective clusters
from collections import Counter
print(Counter(kmeans.labels_))
print(kmeans.cluster_centers_)



ClusterNumbers = pd.DataFrame(kmeans.labels_)
df['ClusterNumbers'] = ClusterNumbers
df = df[['ClusterNumbers', 'Balance', 'Qual_miles', 'cc1_miles', 'cc2_miles', 'cc3_miles','Bonus_miles', 'Bonus_trans', 'Flight_miles_12mo', 'Flight_trans_12','Days_since_enroll', 'Award?']]
#df.insert(1, "Univ", df['Univ']) 
print(df.head(10))

#adding the output to a csv file
df.to_csv('C:/Users/kdandebo/Desktop/Models/DS training/Yogesh data sets/Airline-kmeans.csv')

##Inferences of non-heirarchial clustering is 

#1. cluster number 32 has most number of elements. - 453
#2. No. of iterations before deciding the final model - 300