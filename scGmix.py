import numpy as np
import scanpy as sc
import anndata as adata
import pickle

class scGmix():
    """Single Cell Gaussian mixture model pipeline
    Instance Methods:
    ----------
    `preprocess`
    """
    def __init__(self,adata, _model = None):
        self.adata = adata
        self._model = _model
        row,cols = self.adata.shape

#### Utility ####################################################################################################
    def savefile(self,filenamepath):
      self.adata.write(filenamepath)

    def savemodel(self,filenamepath):
      with open(filenamepath, "wb") as file:
          pickle.dump(self.m_model, file)

#### STAGE 1: PREPROCESSING INSTANCE METHOD ########################################################################
    def preprocess(self, mads_away = 5,feature_selection = False, min_mean=0.0125, max_mean=3, min_disp=0.5):
      """workflow for preprocessing:
      1. Appends qc data on the adata object inplace to be used in 2. , 3.
      2. Handles cell and gene filtering, cells are removed when their log gene counts, or
      log total expression count are above or below `mads_away` (absolute deviations away from the median) in both
      mentioned distributions. Gene are removed if they have 0 gene expression for all cells
      3. Normalizes the counts and log transforms them after adding the value of 1
      4. (OPTIONAL): scanpy's highly variable gene selection based on `min_mean` ,`max_mean`, `min_disp`
      ignored if `feature_selection = False`"""
      # PRIVATE sub-methods ############################
      def _qcMetrics():
        """Void functin that appends qc data on the adata object inplace"""
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

      # function execution ############################
      _qcMetrics() # QC 
      _filter() # QC
      _normalize() # normalization, log transformation
      # optinal higly variable gene selection
      if feature_selection:
        sc.pp.highly_variable_genes(self.adata, min_mean=0.0125, max_mean=3, min_disp=0.5)

#### STAGE 2: Dimensionality reduction ####################################################
    def dimreduction():
       pass




### MAIN #####
adata = sc.read_csv("datasets/dataset1.csv")
pipeline = scGmix(adata) # Creating an instance of Preprocess class
pipeline.preprocess(mads_away=3,feature_selection=False)
pipeline.savefile("test.ata")
print(pipeline.adata)