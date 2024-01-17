from scipy.stats import qmc
import numpy as np
from pathlib import Path

prior = 'vanilla'

if prior == 'baseline':
    header = "logM_cut, logM_1, sigma, alpha, kappa, s, B_cen, B_sat"
    pmins = np.array([12.4, 13.2, -3.0, 0.7, 0.0, -1.0, -1.0, -1.0])
    pmaxs = np.array([13.3, 14.4, 0.0, 1.5, 1.5, 1.0, 1.0, 1.0])
elif prior == 'vanilla':
    header = "logM_cut, logM_1, sigma, alpha, kappa"
    pmins = np.array([12.4, 13.2, -3.0, 0.7, 0.0])
    pmaxs = np.array([13.3, 14.4, 0.0, 1.5, 1.5])

sampler = qmc.LatinHypercube(d=len(pmins), seed=42)
params = sampler.random(n=85000)
params = pmins + params * (pmaxs - pmins)

cosmos = list(range(0, 5)) + list(range(13, 14)) + list(range(100, 127)) + list(range(130, 182))
split_params = np.array_split(params, len(cosmos))

for i, cosmo in enumerate(cosmos):
    output_dir = Path(f'./hod_params/{prior}')
    output_fn = output_dir / f'hod_params_{prior}_c{cosmo:03}.csv'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    np.savetxt(output_fn, split_params[i], header=header, delimiter=',', fmt="%.5f")
