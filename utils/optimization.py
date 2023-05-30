import optuna
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from kneed import KneeLocator
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score , calinski_harabasz_score
optuna.logging.set_verbosity(optuna.logging.WARNING)

def findKmeanselbow(X, min_clusters, max_clusters,seed = 42,show = False):
    """
    Find the optimal number of clusters using the elbow method with k-means clustering.
        X (matrix): The input data.
        min_clusters (int): The minimum number of clusters to consider.
        max_clusters (int): The maximum number of clusters to consider.
    Returns:
        wcss (array): the wcss values for each k
        knee (int): The optimal number of clusters
    """
    n_clusters =  np.arange(min_clusters, max_clusters)
    wcss = np.zeros_like(n_clusters)
    # Perform k-means clustering for each number of clusters
    for i, k in enumerate(n_clusters):
      kmeans = KMeans(n_clusters=k, random_state=seed)
      kmeans.fit(X)
      wcss[i] = kmeans.inertia_
    kneedle = KneeLocator(x =n_clusters, y = wcss, S=1.0, curve="convex", direction="decreasing")
    knee = round(kneedle.knee)
    if show:
      sns.lineplot(x = n_clusters, y =wcss,marker='o')
      plt.xlabel("Number of clusters")
      plt.axvline(x=knee, color = 'red',label = "Elbow point",linestyle="dashed")
      plt.ylabel("Within-Cluster-Sum of Squared Errors")
      plt.title("Elbow plot")
      plt.legend()
      plt.grid()
      plt.show()
    return wcss, knee

def optimizeSpectral(X, k_range: list, neighbors_range :  list, affinity, eval_metric, direction, n_trials=100):
    """
    Optimizes the parameters for spectral clustering using Optuna.
        X (matrix): The input data matrix.
        k_range (list): A list containing two values indicating the minimum and maximum number of clusters to try.
        neighbors_range (list): A list containing two values indicating the minimum and maximum number of neighbors to try.
        affinity (str): The affinity parameter for spectral clustering. Supported values: "laplacian", "rbf", "nearest_neighbors".
        eval_metric (str): The evaluation metric to use. Supported values: "silhouette_score", "davies_bouldin", "calinski_harabasz".
        direction (str): The direction of optimization. Supported values: "minimize", "maximize".
        n_trials (int, optional): The number of optimization trials to perform. Defaults to 100.
    Returns:s
    best_params (array): The best parameters
    best_scores (array): the best values
    study (object): the study object
        """
    def Spectral_trial(trial):
        # Define the parameter search space
        if (len(k_range) == 2) and (len(neighbors_range) == 2):

          if (affinity == "rbf"):
            n_clusters = trial.suggest_int("n_clusters", k_range[0], k_range[1])
            gamma = trial.suggest_float("gamma", 0.01, 1.0)
            spectral = SpectralClustering(n_clusters=n_clusters,
                                          affinity=affinity,
                                          gamma=gamma,
                                          assign_labels="cluster_qr",
                                          random_state=42)
            
          elif(affinity == "nearest_neighbors"):
            n_clusters = trial.suggest_int("n_clusters", k_range[0], k_range[1])
            n_neighbors = trial.suggest_int("n_neighbors",neighbors_range[0], neighbors_range[1])
            spectral = SpectralClustering(n_clusters=n_clusters,
                                          affinity=affinity,
                                          n_neighbors=n_neighbors,
                                          assign_labels="cluster_qr",
                                          random_state=42)
          else: 
            raise ValueError("Unknown affinity parameter, try laplacian, rbf, nearest_neighbors")
        else:
          raise ValueError("k_range, neighbors_range need two items, indicating the min and max clusters, neighbors to try")

        # Find the cluster labels
        labels = spectral.fit_predict(X)
        # Begin evaluation based on the selected metric
        if eval_metric == "silhouette_score":
          try:
              return silhouette_score(X, labels, random_state=42)
          except ValueError:
              # Return the worst score if it does not converge
              return -1
        elif eval_metric == "davies_bouldin":
          try:
              return davies_bouldin_score(X, labels)
          except ValueError:
              # Return the worst score if it does not converge
              return np.inf
        elif eval_metric == "calinski_harabasz":
            return calinski_harabasz_score(X, labels)
    # Trial pruner
    pruner = optuna.pruners.HyperbandPruner()
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)
    study.optimize(Spectral_trial, n_trials=n_trials, show_progress_bar=True)

    # Get the best parameters and objective value
    best_params = study.best_params
    best_value = study.best_value

    return best_params, best_value, study


def optimizeGMM(X,metric = "BIC",components_range =[],precomputed_means=None,n_trials=100):
    """
Args:
    X (matrix): The input data matrix.
    metric (str, optional): The optimization metric to use. Supported values: "BIC", "AIC". Defaults to "BIC".
    components_range (list, optional): A list containing two values indicating the minimum and maximum number of components to try. Defaults to an empty list. Replaces the need for precomputed_means
    precomputed_means (array, optional): Precomputed means for initializing the Gaussian Mixture Model. Defaults to None. Replaces the need for components_range
    n_trials (int, optional): The number of optimization trials to perform. Defaults to 100
Returns:
    best_params (array): The best parameters found during optimization.
    best_value (float): The best objective value (metric score) found during optimization.
    study (object): The study object containing optimization results.
"""
    def bicTrial(trial):
      # Define the parameter search space
      if (len(components_range) != 0):
        n_components = trial.suggest_int("n_components", components_range[0], components_range[1])
        covariance_type = trial.suggest_categorical("covariance_type", ["full", "tied", "diag", "spherical"])
        random_state = trial.suggest_int("random_state",1,n_trials)
        init_params = trial.suggest_categorical("init_params",["kmeans", "k-means++","random","random_from_data"])
        gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type,init_params=init_params,random_state = random_state,max_iter=1000)
      elif (len(components_range) == 0):
        covariance_type = trial.suggest_categorical("covariance_type", ["full", "tied", "diag", "spherical"])
        random_state = trial.suggest_int("random_state",1,n_trials)
        gmm = GaussianMixture(n_components = precomputed_means.shape[0],means_init=precomputed_means, covariance_type=covariance_type,random_state = random_state, max_iter=100)
      else:
          raise ValueError("Invalid parameters, optimizeGMM requires either components_range or precomputed_means")
      # Initialize the model
      # Fit Gaussian Mixture Model
      try:
        gmm.fit(X)
      except ValueError:
        return np.inf
      if metric == "BIC":
        bic_score = gmm.bic(X)
        return bic_score
      elif metric == "AIC":
        aic_score = gmm.aic(X)
        return aic_score
      else:
          raise ValueError ("Invalid metric, choose between BIC, AIC")

    # Define the Optuna study and optimize the objective
    pruner = optuna.pruners.MedianPruner()
    sampler = optuna.samplers.RandomSampler(seed=40)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(bicTrial, n_trials=n_trials, show_progress_bar=True)

    # Get the best parameters and objective value
    best_params = study.best_params
    best_value = study.best_value

    return best_params, best_value, study