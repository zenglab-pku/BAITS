import numpy as np
import pandas as pd
from torchgmm.bayes import GaussianMixture as TorchGaussianMixture
from torchgmm.bayes.gmm.lightning_module import GaussianMixtureLightningModule
from torchgmm.bayes.gmm.model import GaussianMixtureModel
import logging
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from pytorch_lightning import Trainer
from tqdm import tqdm
import torch
from sklearn.metrics import mean_absolute_percentage_error
import anndata as ad
from copy import deepcopy
import scipy.sparse as sps
from typing import List, Tuple, cast, Union
AnyRandom = Union[int, np.random.RandomState, None]
from torchgmm.base.data import (
    DataLoader,
    TensorLike,
    collate_tensor,
    dataset_from_tensors,
)


class GaussianMixture(TorchGaussianMixture):

    model_: GaussianMixtureModel
    #: The average per-datapoint negative log-likelihood at the last training step.

    def __init__(
        self,
        n_clusters: int = 1,
        covariance_type: str = "full",
        init_strategy: str = "kmeans",
        init_means: torch.Tensor = None,
        convergence_tolerance: float = 0.001,
        covariance_regularization: float = 1e-06,
        batch_size: int = None,
        trainer_params: dict = None,
        random_state: AnyRandom = 0, #* 注意这个random_state
    ):
        super().__init__(
            num_components=n_clusters,
            covariance_type=covariance_type,
            init_strategy=init_strategy,
            init_means=init_means,
            convergence_tolerance=convergence_tolerance,
            covariance_regularization=covariance_regularization,
            batch_size=batch_size,
            trainer_params=trainer_params,
        )
        self.n_clusters = n_clusters
        self.random_state = random_state

    def score(self, data: TensorLike):
        return super().score(data) 

    
    def fit(self, data: TensorLike):
        if sps.issparse(data):
            raise ValueError(
                "Sparse data is not supported. You may have forgotten to reduce the dimensionality of the data. Otherwise, please convert the data to a dense format."
            )
        return self._fit(data)

    def _fit(self, data):
        try:
            return super().fit(data)
        except torch._C._LinAlgError as e:
            if self.covariance_regularization >= 1:
                raise ValueError(
                    "Cholesky decomposition failed even with covariance regularization = 1. The matrix may be singular."
                ) from e
            else:
                self.covariance_regularization *= 10
                logger.warning(
                    f"Cholesky decomposition failed. Retrying with covariance regularization {self.covariance_regularization}."
                )
                return self._fit(data)

    def predict(self, data: TensorLike):
        """ 
        Computes the most likely components for each of the provided datapoints.
        
        Attention
        ----------
            When calling this function in a multi-process environment, each process receives only
            a subset of the predictions. If you want to aggregate predictions, make sure to gather
            the values returned from this method.
        """
        return super().predict(data).numpy()

    def predict_proba(self, data: TensorLike):
        """
        Computes a distribution over the components for each of the provided datapoints.
        
        Attention
        ----------
            When calling this function in a multi-process environment, each process receives only
            a subset of the predictions. If you want to aggregate predictions, make sure to gather
            the values returned from this method.
        """
        loader = DataLoader(
            dataset_from_tensors(data),
            batch_size=self.batch_size or len(data),
            collate_fn=collate_tensor,
        ) 
        trainer_params = self.trainer_params.copy()
        trainer_params["logger"] = False
        result = Trainer(**trainer_params).predict(GaussianMixtureLightningModule(self.model_), loader)
        return torch.cat([x[0] for x in cast(List[Tuple[torch.Tensor, torch.Tensor]], result)])

    def score_samples(self, data: TensorLike):
        """
        Computes the negative log-likelihood (NLL) of each of the provided datapoints.
        
        Attention
        ----------
            When calling this function in a multi-process environment, each process receives only
            a subset of the predictions. If you want to aggregate predictions, make sure to gather
            the values returned from this method.
        """
        loader = DataLoader(
            dataset_from_tensors(data),
            batch_size=self.batch_size or len(data),
            collate_fn=collate_tensor,
        )
        trainer_params = self.trainer_params.copy()
        trainer_params["logger"] = False
        result = Trainer(**trainer_params).predict(GaussianMixtureLightningModule(self.model_), loader)
        return torch.stack([x[1] for x in cast(List[Tuple[torch.Tensor, torch.Tensor]], result)])




class ClusterAutoK:
    """
    Identify the best candidates for the number of clusters.
    """
    silhouette_scores: np.ndarray

    def __init__(
        self,
        n_clusters,
        max_runs: int = 5,
        convergence_tol: float = 1e-2,
        model_class: type = None,
    ):
        self.n_clusters = (
            list(range(*(max(2, n_clusters[0] - 1), n_clusters[1] + 1)))
            if isinstance(n_clusters, tuple)
            else n_clusters
        )
        self.max_runs = max_runs
        self.convergence_tol = convergence_tol
        self.model_class = model_class if model_class else GaussianMixture
        self.silhouette_scores = []

    def fit(self, adata: ad.AnnData, use_rep: str = "X_cellcharter", verbose: bool = True): 
        """
        Fit the clustering model with a range of cluster numbers and calculate silhouette scores.
        
        Parameters
        ----------
        adata
            AnnData object containing the data to cluster.
        use_rep
            str, the key in `adata.obsm` to use for clustering. Defaults to "X_cellcharter".
        verbose
            bool, whether to display the fitting process. Defaults to True.
        """
        if use_rep not in adata.obsm:
            raise ValueError(f"{use_rep} not found in adata.obsm. If you want to use adata.X, set use_rep=None") 

        X = adata.obsm[use_rep] if use_rep is not None else adata.X 
        
        self.best_models = {} 
        random_state = 0
        previous_silhouette_scores = None
        for i in range(self.max_runs):
            print(f"Iteration {i + 1}/{self.max_runs}")
            run_silhouette_scores = []

            for k in tqdm(self.n_clusters, disable=not verbose or len(self.n_clusters) == 1):
                # Suppress PyTorch warnings
                logging_level = logging.getLogger("lightning.pytorch").getEffectiveLevel()
                logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
                clustering = self.model_class(n_clusters=k, random_state=i + random_state)
                clustering.fit(X)  
                logging.getLogger("lightning.pytorch").setLevel(logging_level)

                # Track best models by negative log-likelihood (nll_)
                if (k not in self.best_models.keys()) or (clustering.nll_ < self.best_models[k].nll_):
                    self.best_models[k] = clustering

                silhouette_avg = silhouette_score(X, clustering.predict(X))
                run_silhouette_scores.append(silhouette_avg)
        
            self.silhouette_scores.append(run_silhouette_scores) 
            print(self.silhouette_scores)
            
            if i > 0:
                if previous_silhouette_scores is not None:
                    silhouette_scores_change = mean_absolute_percentage_error(
                        np.mean(previous_silhouette_scores, axis=0),
                        np.mean(self.silhouette_scores, axis=0),
                    )
                    if silhouette_scores_change < self.convergence_tol:
                        if verbose:
                            print(f"Convergence with a change in silhouette_scores of {silhouette_scores_change} reached after {i + 1} iterations")
                        break 
                previous_silhouette_scores = deepcopy(self.silhouette_scores)


    @property
    def best_k(self) -> int:
        """The number of clusters with the highest silhouette_scores."""
        if self.max_runs <= 1:
            raise ValueError("Cannot compute silhouette_scores with max_runs <= 1")
        silhouette_scores_mean = np.mean(self.silhouette_scores, axis=0) 
        best_idx = np.argmax(silhouette_scores_mean)
        return self.n_clusters[best_idx]

    def predict(self, 
                adata: ad.AnnData,
                use_rep: str = None,
                k: int = None,
                store_labels: bool = False,  
                store_column: str = 'predicted_labels') -> pd.Categorical:
        """
        Predict cluster labels for the data in the given representation and optionally store the labels in `adata.obs`.

        Parameters
        ----------
        adata
            AnnData object containing the dataset. The data to be clustered is accessed from `adata.obsm` or `adata.X`.
        use_rep
            The key in `adata.obsm` to use as the data representation for clustering. 
            If `None`, the method defaults to:
            - `adata.obsm['X_cellcharter']`, if it exists, or
            - `adata.X` as a fallback.
        k
            The number of clusters to predict labels for. If not specified, the best number of clusters (`self.best_k`) 
            will be used. Must be one of the values in `self.n_clusters`.
        store_labels
            If `True`, the predicted labels will be stored in `adata.obs` under the column name specified by `store_column`. 
            Default is `False`.
        store_column
            The name of the column in `adata.obs` where predicted labels will be stored if `store_labels` is `True`. 
            Default is `'predicted_labels'`.
        Returns
        -------
        pd.Categorical
            A pandas Categorical object containing the predicted cluster labels. The labels are integers ranging from 0 to `k-1`.
        Raises
        ------
        AssertionError
            If `k` is provided and it is not in `self.n_clusters`.
            
        Notes
        -----
        - This method relies on the clustering models stored in `self.best_models` for label prediction.
        - Ensure that the model for the desired `k` clusters has been fitted prior to calling this method.
        """
        k = self.best_k if k is None else k
        assert k is None or k in self.n_clusters

        X = (
            adata.obsm[use_rep]
            if use_rep is not None
            else adata.obsm["X_cellcharter"] if "X_cellcharter" in adata.obsm else adata.X
        )

        labels = pd.Categorical(self.best_models[k].predict(X), categories=np.arange(k))
        if store_labels:
            adata.obs[store_column] = labels 

        return labels
