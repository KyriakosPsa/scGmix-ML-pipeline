import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc

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
  plt.xlabel(f"PC1: {round(adata.uns['pca']['variance_ratio'][0]*100,2)}% of total variance ratio")
  plt.ylabel(f"PC2: {round(adata.uns['pca']['variance_ratio'][1]*100,2)}% of total variance ratio")
  plt.grid()
  plt.show()