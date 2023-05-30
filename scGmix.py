# libraries
import numpy as np
import scanpy as sc
import anndata as adata
import matplotlib.pyplot as plt
import seaborn as sns
from kneed import KneeLocator
import pickle
# Consider aesthetics
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['axes.axisbelow'] = True
plt.rcParams.update({'font.size': 18})
sc.set_figure_params(scanpy=True, dpi=80, dpi_save=300, 
                    frameon=True, vector_friendly=True, fontsize=18, figsize=(8,8),
                    format='png', ipython_format='png2x')


class scGmix():
    """Single Cell Gaussian mixture model pipeline
    Instance Methods:
    ----------
    `preprocess`
    `dimreduction`
    """
    def __init__(self,adata, _model = None):
        self.adata = adata
        self._model = _model
        self.row, self.cols = self.adata.shape

#### Utility ####################################################################################################
    def savefile(self,filenamepath):
      self.adata.write(filenamepath)

    def savemodel(self,filenamepath):
      with open(filenamepath, "wb") as file:
          pickle.dump(self.m_model, file)

#### STAGE 1: PREPROCESSING INSTANCE METHOD ########################################################################
    def preprocess(self, mads_away = 5,feature_selection = False, min_mean=0.0125, max_mean=3, min_disp=0.5):
      """
      Performs preprocessing steps on the data.

      Parameters:
      - mads_away (int): Number of median absolute deviations (MADs) away from the median for cell filtering. Default is 5.
      - feature_selection (bool): Whether to perform feature selection using scanpy's highly variable gene selection. Default is False.
      - min_mean (float): The minimum mean expression threshold for highly variable gene selection. Ignored if feature_selection is False. Default is 0.0125.
      - max_mean (float): The maximum mean expression threshold for highly variable gene selection. Ignored if feature_selection is False. Default is 3.
      - min_disp (float): The minimum dispersion threshold for highly variable gene selection. Ignored if feature_selection is False. Default is 0.5.
      """
      # PRIVATE preprocess sub-methods############################
      def _qcMetrics():
        """Void functin that appends qc data on the adata object inplace
        copy of scanpy's sc.pp.calculate_qc_metrics"""
        sc.pp.calculate_qc_metrics(self.adata,
                                  percent_top = None,
                                  log1p= True,
                                  inplace = True)
      
      def _medianAbsdev(qc_metric):
        """Function that returns the median absolute deviation for a QC metric"""
        return np.median(np.abs(qc_metric - np.median(qc_metric)))
      
      def _filter():
        """Void function that handles cell and gene filtering, cells are removed when their log gene counts, or
        log total expression count are above or below mads_away (absolute deviations away from the median) in both
        mentioned distributions. Gene are removed if they have 0 gene expression for all cells
        """
        m1 = self.adata.obs["log1p_n_genes_by_counts"] # metric 1 for cell filtering
        m2 = self.adata.obs["log1p_total_counts"] # metric 2 for cell filtering 
        m3 = self.adata.var["n_cells_by_counts"] # metric 3 for gene filtering
        # cell filtering
        cell_mask = (m1 < np.median(m1) - mads_away * np.median(m1)) | (m1 > np.median(m1) + mads_away * _medianAbsdev(m1) ) &\
        (m2 < np.median(m2) - mads_away * _medianAbsdev(m2)) | (m2 > np.median(m2) + mads_away)
        self.adata = self.adata[~cell_mask]
        # gene filtering
        gene_mask = (m3 == 0) 
        self.adata.var_names = self.adata.var_names[~gene_mask]

      def _normalize():
        """Void function the normalizes the counts and log transforms them after adding the value of 1"""
        # Normalization
        sc.pp.normalize_total(self.adata, target_sum=None, inplace=True)
        # log1p transform
        self.adata.X = sc.pp.log1p(self.adata.X)

      # preprocessing method execution 
      #-----------------------------------------
      _qcMetrics() # QC 
      _filter() # QC
      _normalize() # normalization, log transformation
      # optinal higly variable gene selection
      if feature_selection:
        sc.pp.highly_variable_genes(self.adata, min_mean=0.0125, max_mean=3, min_disp=0.5)

#### STAGE 2: Dimensionality reduction ####################################################
    def dimreduction(self,method = "PCA", n_pcs = 100, pc_selection_method = "screeplot",  n_neighbors=15, min_dist = 0.1,
                  use_highly_variable = False,variance_threshold = 90, verbose = True,plot_result = False):
      """
      Performs dimensionality reduction and optimal component selection on the data using the specified method: PCA, TSNE or UMAP 
      & screeplot plot, variance threshold or kaiser's rule.
      Parameters:
      - method (str): The dimensionality reduction method to use. Options are "PCA", "TSNE", or "UMAP". Default is "PCA".
      - n_pcs (int): Number of principal components to use in the initial pca.
      - pc_selection_method (str): The method to determine the optimal number of principal components. Options are "screeplot", "kaiser", or "variance". Default is "screeplot".
      - n_neighbors (int): The number of neighbors to consider for UMAP. Default is 15., ignored for PCA, TSNE
      - min_dist (float): The minimum distance between points for UMAP. Default is 0.1, ignored for PCA, TSNE.
      - use_highly_variable (bool): Whether to use only highly variable genes for PCA. Default is False.
      - variance_threshold (int): The threshold for variance-based principal component selection. Default is 90, ignored if "knee", or "kaiser".
      - verbose (bool): Wether or not to print the optimal number of components found.
      - plot_result (bool):  Wether or not to plot the results of the dimensionality reduction.
      """
      # PRIVATE optimal number of principal components selection methods
      def _kneemethod(explained_variance):
        """Screeplot plot knee method, knee point identified via the kneed library"""
        kneedle = KneeLocator(x =np.arange(1,explained_variance.shape[0]+1,1), y = explained_variance, S=1.0, curve="convex", direction="decreasing")
        optimal_pcs = explained_variance[:round(kneedle.knee)]
        return optimal_pcs
      
      def _variancethreshold(explained_variance,threshold):
        """cummulative variance threshold"""
        cummulative_variance = np.cumsum(explained_variance)
        optimal_pcs = cummulative_variance[cummulative_variance <= threshold]
        return optimal_pcs
      
      def _kaiserule(explained_variance):
        """kaiser's rule threshold"""
        optimal_pcs = explained_variance[explained_variance > 1]
        return optimal_pcs
      
      def _compute():
        """Compute the dimensionality reduction representation and the optimal number of components"""
        # Pca is needed as initilization for TSNE, UMAP even if its not picked as the method the user chooses
        sc.pp.pca(self.adata, svd_solver ='arpack',use_highly_variable = use_highly_variable, n_comps = n_pcs)
        explained_variance = self.adata.uns['pca']['variance']
        # Check for the selected principal componenet selection method
        if pc_selection_method == "screeplot":
          optimal_pcs = _kneemethod(explained_variance)
        elif pc_selection_method == "kaiser":
          optimal_pcs = _kaiserule(explained_variance)
        elif pc_selection_method == "variance":
          optimal_pcs = _variancethreshold(explained_variance,variance_threshold)
        else:
          raise ValueError("Invalid pc selection method, choose between screeplot, kaiser, variance")
        if verbose:
            print(f"{pc_selection_method} selected {optimal_pcs.shape[0]} principal components out of {n_pcs}")
        # Check the dimensionality reduction method
        if method == "PCA":
          self.adata.uns['pca']['variance'] = self.adata.uns['pca']['variance'][optimal_pcs.shape[0]]
          self.adata.obsm['X_pca'] = self.adata.obsm['X_pca'][:,:optimal_pcs.shape[0]]
        elif method == "TSNE":
          # Run tsne with the suggested paramters from [The art of using t-SNE for single-cell transcriptomics]
          n = self.row/100
          if n/100 > 30:
            perplexity = 30 + n/100 
          else:
            perplexity = 30
          if n > 200:
            learning_rate = n/12
          else:
            learning_rate = 200
          sc.tl.tsne(self.adata, 
                    n_pcs = optimal_pcs.shape[0], 
                    perplexity = perplexity, 
                    early_exaggeration=12, 
                    learning_rate = learning_rate, 
                    random_state = 42, 
                    use_fast_tsne = False)
        elif method  == "UMAP":
          sc.pp.neighbors(self.adata, n_neighbors=n_neighbors, random_state=42, n_pcs= optimal_pcs.shape[0])
          sc.tl.umap(self.adata, min_dist= min_dist, random_state=42)
        else:
          raise ValueError("Invalid dimensionality reduction method, choose between PCA, TSNE, UMAP")

      def _plot(self):
        """plot the results"""
        if method == "PCA":
          sc.pl.pca(self.adata,color="total_counts",add_outline = True, size = 100, title= "PCA, cell total expression counts colormap",show=False)
          plt.xlabel(f"PC1: {round(self.adata.uns['pca']['variance_ratio'][0]*100,2)}% of total variance")
          plt.ylabel(f"PC2: {round(self.adata.uns['pca']['variance_ratio'][1]*100,2)}% of total variance")
          plt.show()
        elif method == "TSNE":
          sc.pl.tsne(self.adata,add_outline = True, size = 100,title = "t-SNE, cell total expression counts colormap",color="total_counts")
        elif method  == "UMAP":
          sc.pl.umap(self.adata, add_outline=True, size=100, title="UMAP, cell total expression counts colormap", color="total_counts")

      # dimensionality reduction method execution 
      #-----------------------------------------
      _compute()
      if plot_result:
        _plot(self)

### MAIN #####
adata = sc.read_csv("datasets/dataset1.csv")
pipeline = scGmix(adata) # Creating an instance of Preprocess class
pipeline.preprocess(mads_away=5,feature_selection=False)
pipeline.dimreduction(plot_result=True,method="UMAP", pc_selection_method="kaiser")