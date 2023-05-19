import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc

def hist_subplot(data1,data2,xlabel1,xlabel2,title1,title2):
  """
  Create a figure with two subplots displaying distribution plots 
  (displot) of the given data and their median values.
  """
  median1 = data1.median()
  median2 = data2.median()

  fig, axs = plt.subplots(1, 2, figsize=(12, 6))

  sns.histplot(data1, kde=True, bins=30, ax=axs[0])
  axs[0].set_title(title1)
  axs[0].axvline(x=median1, color='red', linestyle='--', label='Median')
  axs[0].grid()
  axs[0].set_xlabel(xlabel1)
  axs[0].legend()
  sns.histplot(data2, kde=True, bins=30, ax=axs[1])
  axs[1].set_title(title2)
  axs[1].axvline(x=median2, color='red', linestyle='--', label='Median')
  axs[1].grid()
  axs[1].set_xlabel(xlabel2)
  axs[1].legend()

  plt.tight_layout()
  fig.show()