#K-MEANS FUNCTION
#import

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from kneed import KneeLocator
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from yellowbrick.cluster import SilhouetteVisualizer
import seaborn as sns
import warnings
from sklearn.cluster import DBSCAN # To instantiate and fit the model
from sklearn.metrics import pairwise_distances # For Model evaluation
from sklearn.neighbors import NearestNeighbors # For Hyperparameter Tuning
from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

#PATH dove andare a leggere e a scrivere
PATH_TO_WRITE_ETL = "./dataset_modificato_WC/"
PATH_FROM_READ_ETL = "./dataset_originale_WC/"
PATH_FROM_READ_ANALYSIS = "./dataset_modificato_WC/" 


#create dict
def create_dict(dimension):
    dict = {}
    for i in range(0,dimension):
        dict[i] = str(i)
    return dict

#normalizzazione
def normalize_df(df_to_normalize):
    scaler = StandardScaler()
    df_std = scaler.fit_transform(df_to_normalize)
    print(np.mean(df_std),np.std(df_std))
    print(df_std.shape)
    return df_std

#plot varianca ratio
def plot_variance_ratio(df_std):
    pca = PCA()
    pca.fit(df_std)
    pca.explained_variance_ratio_

    plt.figure(figsize=(10,8))
    plt.plot(range(1,df_std.shape[1]+1),pca.explained_variance_ratio_.cumsum(),marker='o',linestyle='--')
    plt.title('Varianza tra le componenti')
    plt.xlabel('Numero di componenti')
    plt.ylabel('Varianza comulativa')

#apply pca method
def apply_pca(df_std,n_components):
    pca = PCA(n_components = n_components, svd_solver='full')
    pca.fit(df_std)
    scores_pca = pca.transform(df_std)
    return scores_pca

#get wcss for elbow method
def get_wcss(scores_pca):
    wcss = []
    for i in range(1,50):
        kmeans_pca = KMeans(n_clusters=i,init='k-means++',random_state=42)
        kmeans_pca.fit(scores_pca)
        wcss.append(kmeans_pca.inertia_)
    return wcss

#plot of wcss score for Elbow Method and apply k means
def plot_wcss_and_apply_kMeans(wcss,scores_pca):
    plt.figure(figsize=(10,8))
    plt.plot(range(1,10),wcss,marker='o',linestyle='--')
    plt.title('K-Means utilizzando PCA')
    plt.xlabel('Numero di cluster')
    plt.ylabel('WCSS')
    kl = KneeLocator(range(1,50),wcss,curve="convex",direction="decreasing")
    print("Numero di cluster (k-means):",kl.elbow)
    kmeans_pca = KMeans(n_clusters=kl.elbow,init='k-means++',random_state=42)
    kmeans_pca.fit(scores_pca)
    return kmeans_pca

def concat_df(original_df,scores_pca,kmeans_pca):
    df_segm_pca_kmeans = pd.concat([original_df.reset_index(drop=True),pd.DataFrame(scores_pca)],axis=1)
    df_segm_pca_kmeans.columns.values[-abs(scores_pca.shape[1]):] = ["C-" + str(s+1) for s in create_dict(scores_pca.shape[1]).keys()]
    df_segm_pca_kmeans['Segm K-means PCA'] = kmeans_pca.labels_
    df_segm_pca_kmeans['Segment'] = df_segm_pca_kmeans['Segm K-means PCA'].map(create_dict(len(kmeans_pca.labels_)))
    df_segm_pca_kmeans.head()
    return df_segm_pca_kmeans

#plot results on 2 dimension
def view_kmeans_results_2d(df_segm_pca_kmeans,scores_pca,kmeans_pca):
    x_axis=df_segm_pca_kmeans['C-1']
    y_axis=df_segm_pca_kmeans['C-2']
    plt.figure(figsize=(10,8))
    sns.scatterplot(x=x_axis,y=y_axis,hue=df_segm_pca_kmeans['Segment'],palette='colorblind')
    #plt.scatter(kmeans_pca.cluster_centers_[:,0], kmeans_pca.cluster_centers_[:, 1], s = 50, c = 'blue' , label = 'centroid')
    plt.title('Clusters')
    plt.show()
    return df_segm_pca_kmeans

#evaluation
def evaluation(scores_pca,kmeans_pca):
    kmeans_silhuouette = silhouette_score(scores_pca,kmeans_pca.labels_).round(2)
    print(kmeans_silhuouette)
    visualizer = SilhouetteVisualizer(kmeans_pca, colors='yellowbrick')

    plt.figure(figsize=(15,8))
    visualizer.fit(scores_pca)        # Fit the data to the visualizer
    visualizer.show()        # Finalize and render the figure

#save results
def save_clusters_in_csv_files(df_segm_pca_kmeans,scores_pca,path):
    for i in range(0,max(df_segm_pca_kmeans['Segment'].astype(int))+1):
        df = df_segm_pca_kmeans[df_segm_pca_kmeans["Segm K-means PCA"]==i]
        df.drop(["C-" + str(s+1) for s in create_dict(scores_pca.shape[1]).keys()],axis=1,inplace=True)
        df.to_csv(path +str(i)+".csv",index=False)

#calculate clusters centroids
def calculate_centroids(df_segm_pca_kmeans,scores_pca,path,n_first_column_to_delete):
    df2 = pd.DataFrame()
    for i in range(0,max(df_segm_pca_kmeans['Segment'].astype(int))+1):
        df = df_segm_pca_kmeans[df_segm_pca_kmeans["Segm K-means PCA"]==i]
        df.drop(["C-" + str(s+1) for s in create_dict(scores_pca.shape[1]).keys()],axis=1,inplace=True)
        df = df.iloc[: , :-2]
        df = df.iloc[:,n_first_column_to_delete:]
        normalize_df(df)
        average = df.mean()
        average["group"] = i
        total = df.count()
        average.name = 'mean'
        df2 = df2.append(average)
    return df2