import optuna
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
optuna.logging.set_verbosity(optuna.logging.WARNING)

def optimizeSpectral(X, k_range: list, neighbors_range :  list, n_trials=100):
    def Spectral_trial(trial):
        # Define the parameter search space
        if (len(k_range) == 2):
          n_clusters = trial.suggest_int("n_clusters", k_range[0], k_range[1])
        else:
          n_clusters = k_range[0]
        n_neighbors = trial.suggest_int("n_neighbors", neighbors_range[0],neighbors_range[1])
        clustering = SpectralClustering(n_clusters=n_clusters, affinity="nearest_neighbors",n_neighbors=n_neighbors,assign_labels="cluster_qr")
        # Find the cluster labels
        labels = clustering.fit_predict(X)
        try:
            return silhouette_score(X, labels, random_state=42)
        except ValueError:
            # Return the worst score if it does not converge
            return -1

    pruner = optuna.pruners.HyperbandPruner()
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(Spectral_trial, n_trials=n_trials, show_progress_bar=True)

    # Get the best parameters and objective value
    best_params = study.best_params
    best_value = study.best_value

    return best_params, best_value, study



def optimizeKmeans(X,k_range :list, n_trials=100):
    def Kmeans_trial(trial):
      # Define the parameter search space
      if (len(k_range)==2):
        n_clusters = trial.suggest_int("n_clusters", k_range[0], k_range[1])
      else:
        n_clusters =  k_range[0]
      random_state = random_state = trial.suggest_int("random_state", 0, n_trials)
      kmeans = KMeans(n_clusters=n_clusters,random_state=random_state)
      # Find the cluster numbers
      labels = kmeans.fit_predict(X)
      try:
        return silhouette_score(X,labels,random_state=42)
      except ValueError:
        # Return the worst score if it does not converge
        return -1

    pruner = optuna.pruners.MedianPruner()
    sampler = optuna.samplers.RandomSampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(Kmeans_trial, n_trials=n_trials, show_progress_bar=True)

    # Get the best parameters and objective value
    best_params = study.best_params
    best_value = study.best_value

    return best_params, best_value, study


def optimizeAgglomerative(X, k_range: list, n_trials=100):
    def Ward_trial(trial):
        # Define the parameter search space
        n_clusters = trial.suggest_int("n_clusters", k_range[0], k_range[1])
        linkage = trial.suggest_categorical("linkage", ["ward", "complete", "average", "single"])
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        # Find the cluster labels
        labels = clustering.fit_predict(X)
        try:
            return mutual_info_score(X, labels, )
        except ValueError:
            # Return the worst score if it does not converge
            return -1

    pruner = optuna.pruners.HyperbandPruner()
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(Ward_trial, n_trials=n_trials, show_progress_bar=True)

    # Get the best parameters and objective value
    best_params = study.best_params
    best_value = study.best_value

    return best_params, best_value, study


def optimizeGMM(X, components_range : list,best_kmeans_seed, n_trials=100):
    """
    This function optimizes the parameters of a Gaussian Mixture Model using Optuna.
    It searches for the best number of components and covariance matrix structure based 
    on the average of Bayesian Information Criterion (BIC) plus Akaike Information Criterion (AIC) scores.
    """  
    def bicTrial(trial):
      # Define the parameter search space
      n_components = trial.suggest_int("n_components", components_range[0], components_range[1])
      covariance_type = trial.suggest_categorical("covariance_type", ["full", "tied", "diag", "spherical"])
      gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type,init_params='kmeans',random_state = best_kmeans_seed)
      # Fit Gaussian Mixture Model
      try:
        gmm.fit(X)
        bic_score = gmm.bic(X)
        return bic_score
      except ValueError:
        return np.inf

    # Define the Optuna study and optimize the objective
    pruner = optuna.pruners.HyperbandPruner()
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(bicTrial, n_trials=n_trials, show_progress_bar=True)

    # Get the best parameters and objective value
    best_params = study.best_params
    best_value = study.best_value

    return best_params, best_value, study