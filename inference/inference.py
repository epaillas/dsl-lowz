from abc import ABC, abstractmethod
from pathlib import Path
import importlib
import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
import matplotlib.pyplot as plt


class Inference(ABC):
    def __init__(
        self,
        theory_model,
        observation: np.array,
        covariance_matrix: np.array,
        priors: Dict,
        fixed_parameters: Dict[str, float],
        output_dir: Path,
        device: str = "cpu",
    ):
        """Given an inference algorithm, a theory model, and a dataset, get posteriors on the
        parameters of interest. It assumes a gaussian likelihood.

        Args:
            theory_model (callable): model used to predict the observable
            observation (np.array): observed data
            covariance_matrix (np.array): covariance matrix of the data
            priors (Dict): prior distributions for each parameter
            fixed_parameters (Dict[str, float]): dictionary of parameters that are fixed and their values
            output_dir (Path): directory where results will be stored
            device (str, optional): gpu or cpu. Defaults to "cpu".
        """
        self.theory_model = theory_model
        self.observation = observation
        self.covariance_matrix = covariance_matrix
        self.inverse_covariance_matrix = self.invert_covariance(
                covariance_matrix=self.covariance_matrix,
            )
        parameters_to_fit = [
            p for p in theory_model.parameters if p not in fixed_parameters.keys()
        ]
        print(parameters_to_fit)
        self.priors = self.get_priors(priors, parameters_to_fit)
        self.n_dim = len(self.priors)
        self.fixed_parameters = fixed_parameters
        self.device = device
        self.param_names = list(self.priors.keys())
        self.output_dir = Path(output_dir)

    
    def get_priors(
        cls, prior_config: Dict[str, Dict], parameters_to_fit: List[str]
    ) -> Dict:
        """Initialize priors for a given configuration and a list of parameters to fit

        Args:
            prior_config (Dict[str, Dict]): configuration of priors
            parameters_to_fit (List[str]): list of parameteters that are being fitted

        Returns:
            Dict: dictionary with initialized priors
        """
        distributions_module = importlib.import_module(prior_config.pop("stats_module"))
        prior_dict = {}
        for param in parameters_to_fit:
            config_for_param = prior_config[param]
            prior_dict[param] = cls.initialize_distribution(
                distributions_module, config_for_param
            )
        return prior_dict

    @classmethod
    def initialize_distribution(
        cls, distributions_module, dist_param: Dict[str, float]
    ):
        """Initialize a given prior distribution fromt he distributions_module

        Args:
            distributions_module : module form which to import distributions
            dist_param (Dict[str, float]): parameters of the distributions

        Returns:
            prior distirbution
        """
        if dist_param["distribution"] == "uniform":
            max_uniform = dist_param.pop("max")
            min_uniform = dist_param.pop("min")
            dist_param["loc"] = min_uniform
            dist_param["scale"] = max_uniform - min_uniform
        if dist_param["distribution"] == "norm":
            mean_gaussian = dist_param.pop("mean")
            dispersion_gaussian = dist_param.pop("dispersion")
            dist_param["loc"] = mean_gaussian
            dist_param["scale"] = dispersion_gaussian
        dist = getattr(distributions_module, dist_param.pop("distribution"))
        return dist(**dist_param)


    @abstractmethod
    def __call__(
        self,
    ):
        pass

    def invert_covariance(
        self,
        covariance_matrix: np.array,
    ) -> np.array:
        """invert covariance matrix

        Args:
            covariance_matrix (np.array): covariance matrix to invert

        Returns:
            np.array: inverse covariance
        """
        return np.linalg.inv(covariance_matrix)

    def get_covariance_correction(
        n_s, n_d, n_theta=None, correction_method='percival',
    ):
        if correction_method == 'percival':
            B = (n_s - n_d - 2) / ((n_s - n_d - 1)*(n_s - n_d - 4))
            factor = (n_s - 1)*(1 + B*(n_d - n_theta))/(n_s - n_d + n_theta - 1)
        elif correction_method == 'hartlap':
            factor = (n_s - 1)/(n_s - n_d - 2)
        else:
            factor = 1.0
        return factor

    def get_loglikelihood_for_prediction(
        self,
        prediction: np.array,
    ) -> float:
        """Get gaussian loglikelihood for prediction

        Args:
            prediction (np.array): model prediction

        Returns:
            float: log likelihood
        """
        diff = prediction - self.observation
        return -0.5 * diff @ self.inverse_covariance_matrix @ diff

    def get_loglikelihood_for_prediction_vectorized(
        self,
        prediction: np.array,
    ) -> np.array:
        """Get vectorized loglikelihood prediction

        Args:
            prediction (np.array): prediciton in batches

        Returns:
            np.array: array of likelihoods
        """
        diff = prediction - self.observation
        right = np.einsum("ik,...k", self.inverse_covariance_matrix, diff)
        return -0.5 * np.einsum("ki,ji", diff, right)[:, 0]


    def sample_parameters_from_prior(
        self,
    ):
        params = {}
        for param, dist in self.priors.items():
            params[param] = dist.rvs()
        for p, v in self.fixed_parameters.items():
            params[p] = v
        return params

    def sample_from_prior(
        self,
    ) -> Tuple:
        """Sample predictions from prior

        Returns:
            Tuple: tuple of parameters and theory model predictions
        """
        params = self.sample_parameters_from_prior()
        return params, self.theory_model(
            params,
        )

    def get_model_prediction(
        self,
        parameters: np.array,
    ) -> np.array:
        """Get model prediction for a given set of input parameters

        Args:
            parameters (np.array): input parameters

        Returns:
            np.array: model prediction
        """
        params = dict(zip(list(self.priors.keys()), parameters))
        for i, fixed_param in enumerate(self.fixed_parameters.keys()):
            params[fixed_param] = self.fixed_parameters[fixed_param]
        for key in params.keys():
            params[key] = [params[key]]
        return self.theory_model.predictions_np(
            params,
        )[0]