import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from matplotlib.colors import LogNorm

def hist_subplot(data1,data2,xlabel1,xlabel2,title1,title2,plot_median = False):
  """
  Create a figure with two subplots displaying distribution plots.
  """
  if plot_median:
    median1 = data1.median()
    median2 = data2.median()
  fig, axs = plt.subplots(1, 2, figsize=(12, 6))
  sns.histplot(data1, kde=True, bins=30, ax=axs[0])
  axs[0].set_title(title1)
  if plot_median:
    axs[0].axvline(x=median1, color='red', linestyle='--', label='Median')
    axs[0].legend()
  axs[0].grid()
  axs[0].set_xlabel(xlabel1)
  sns.histplot(data2, kde=True, bins=30, ax=axs[1])
  axs[1].set_title(title2)
  if plot_median:
    axs[1].axvline(x=median2, color='red', linestyle='--', label='Median')
    axs[1].legend()
  axs[1].grid()
  axs[1].set_xlabel(xlabel2)
  plt.tight_layout()
  fig.show()

def plot_pca(adata,title = "PCA", save = False):
  """
  Create a figure of the the two PCs after pca preprocessing on the adata object using scanpy
  """
  plt.figure(figsize=(8,8))
  plt.title(title)
  plt.scatter(x = adata.obsm['X_pca'][:,0],y = adata.obsm['X_pca'][:,1], edgecolors='black', alpha= 0.8)
  plt.xlabel(f"PC1: {round(adata.uns['pca']['variance_ratio'][0]*100,2)}% of total variance")
  plt.ylabel(f"PC2: {round(adata.uns['pca']['variance_ratio'][1]*100,2)}% of total variance")
  plt.grid()
  plt.show()

def plot_pca_variance(variance_ratio,n_components,variance_cutoff = 0.90,verbose = True):
  cummulative_variance = np.cumsum(variance_ratio)
  plt.figure(figsize=(8,8))
  sns.lineplot(x = np.arange(1,n_components+1,1), y =cummulative_variance,marker='o')
  plt.xlabel("Principal Components")
  plt.axhline(y=variance_cutoff, color = 'red',label = "90% of the total variance",linestyle="dashed")
  plt.ylabel("Cummulative Variance explained")
  plt.title("CDF of the explained variance of the PCs")
  plt.yticks(np.arange(0,1.1,0.1))
  plt.legend()
  plt.grid()
  plt.show()
  #Show the PCs kept with this method
  if verbose:
    print(f"Variance Threshold of {variance_cutoff}% keeps: ",(cummulative_variance[cummulative_variance <= variance_cutoff]).shape[0], "PCs")

def plot_clusters(X, labels):
    plt.figure(figsize=(6, 6))
    plt.title('First two PCs with cluster labeled data')
    
    # Scatter plot of data points
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette="tab10", edgecolors="black")
    
    # Compute mean points for each cluster
    unique_labels = np.unique(labels)
    cluster_means = [np.mean(X[labels == label], axis=0) for label in unique_labels]
    cluster_means = np.array(cluster_means)
    
    # Plot mean points
    plt.scatter(cluster_means[:, 0], cluster_means[:, 1], marker='X', color='black', s=100, label='Cluster mean')
    
    # Adjust plot settings
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid()
    plt.legend(fontsize = 11)
    plt.show()