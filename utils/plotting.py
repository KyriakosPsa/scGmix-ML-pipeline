import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from matplotlib.colors import LogNorm
import matplotlib as mpl

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


def plot_bic(components,values,idx,criterion):
  """
  Create a figure of the BIC values per number of GMM components
  """
  plt.figure(figsize=(8,8))
  sns.lineplot(x = np.arange(1,components+1,1), y =values, markers=".")
  plt.axvline(x = idx, color = 'red',label = "Optimal number of components",linestyle="dashed" )
  plt.title(f"{criterion} values per # of components")
  plt.xlabel(f"Number of components")
  plt.xticks(np.arange(1,components+1,1))
  plt.ylabel(f"{criterion} score")
  plt.legend()
  plt.show()

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

def make_ellipses(gmm, X, labels, title = "PCA"):
    num_components = len(gmm.means_)
    colors = plt.cm.get_cmap('Set2', num_components)
    fig, ax = plt.subplots(figsize = (7,7))
    for n in range(num_components):
        color = colors(n)
        if gmm.covariance_type == "full":
            covariances = gmm.covariances_[n][:2, :2]
        elif gmm.covariance_type == "tied":
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == "diag":
            covariances = np.diag(gmm.covariances_[n][:2])
        elif gmm.covariance_type == "spherical":
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        ell = mpl.patches.Ellipse(
            gmm.means_[n, :2], v[0], v[1], angle=180 + angle, color=color
        )
        ell.set_alpha(0.5)
        ax.add_artist(ell)
        ax.set_aspect("equal", "datalim")
    
    for i,mean in enumerate(gmm.means_):
        color = colors(i)
        data = X[labels == i]
        ax.scatter(data[:, 0], data[:, 1], s=50,edgecolors ="black", color=color, label=f"Component {i}",alpha=0.8)
        ax.scatter(mean[0],mean[1], marker= 'x', s = 100, color = "black",linewidths = 4, edgecolors="white")
    plt.title(title)
    plt.legend()
    if title == "PCA":
      plt.xlabel("PC1")
      plt.ylabel("PC2")
    else:
      plt.xlabel(f"{title}1")
      plt.ylabel(f"{title}2") 
    plt.grid()
    plt.show()
