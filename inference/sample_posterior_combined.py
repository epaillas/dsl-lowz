
import scipy.stats as st
from cosmopower import cosmopower_NN
import argparse
import numpy as np
from antares import setup_logging
from antares.samplers import NestedSampler
from antares.observables import BaseObservable
from antares.likelihoods import ObservablesGaussianLikelihood
from pathlib import Path


setup_logging()
parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str, required=True)
args = parser.parse_args()

priors = {
    'stats_module': 'scipy.stats',
    'omega_b': {'distribution': 'uniform', 'min': 0.0207, 'max': 0.0243},
    'omega_cdm': {'distribution': 'uniform', 'min': 0.1032, 'max': 0.140},
    'sigma8_m': {'distribution': 'uniform', 'min': 0.678, 'max': 0.938},
    'n_s': {'distribution': 'uniform', 'min': 0.9012, 'max': 1.025},
    'alpha_s': {'distribution': 'uniform', 'min': -0.038, 'max': 0.038},
    'N_ur': {'distribution': 'uniform', 'min': 1.188, 'max': 2.889},
    'w0_fld': {'distribution': 'uniform', 'min': -1.22, 'max': -0.726},
    'wa_fld': {'distribution': 'uniform', 'min': -0.628, 'max': 0.621},
    'logM_1': {'distribution': 'uniform', 'min': 13.2, 'max': 14.4},
    'logM_cut': {'distribution': 'uniform', 'min': 12.4, 'max': 13.3},
    'alpha': {'distribution': 'uniform', 'min': 0.7, 'max': 1.5},
    'sigma': {'distribution': 'uniform', 'min': -3., 'max': 0},
    'kappa': {'distribution': 'uniform', 'min': 0., 'max': 1.5},
    'B_cen': {'distribution': 'uniform', 'min': -1.0, 'max': 1.0},
    'B_sat': {'distribution': 'uniform', 'min': -1., 'max': 1.},
    's': {'distribution': 'uniform', 'min': -1., 'max': 1.},
}

fixed_params = {'alpha_s': 0.0, 'N_ur': 2.0328, 'w0_fld': -1.0, 'wa_fld': 0.0}

# generate a list of observables
observables = []
for name in ['gammat', 'wtheta', 'DSL', 'DSC']:
    data_dir = f'/pscratch/sd/e/epaillas/dsl-lowz'
    data_fn = Path(data_dir) / f'{name}_corrected.npy'
    data = np.load(data_fn)[0]

    theory_fn = Path(data_dir) / f'emu_{name}'
    theory = cosmopower_NN(restore=True, restore_filename=str(theory_fn))

    observable = BaseObservable(data=data, theory=theory)
    observables.append(observable)

# load the covariance of the joint observables
cov_dir = '/pscratch/sd/e/epaillas/dsl-lowz'
cov_fn = Path(cov_dir) / 'total_cov.npy'
covariance = np.load(cov_fn)

likelihood = ObservablesGaussianLikelihood(observables=observables, covariance=covariance)

nested = NestedSampler(
    likelihood=likelihood,
    priors=priors,
    fixed_parameters=fixed_params,
    output_dir=args.output,
)

nested()
