from astropy.io import fits
import numpy as np
from pathlib import Path
import pandas as pd
from pymocker.catalogues import read_utils
import sys
import argparse



if __name__ == '__main__':
    phase = 0
    hods = list(range(0, 100))
    cosmos = [000, 100, 101, 105, 106, 107, 109, 111, 112,
              116, 117, 120, 121, 122, 123, 124, 130, 131,
              132, 133, 134, 135, 136, 137, 138, 139, 140,
              141, 142, 143, 144, 146, 148, 149, 150, 151,
              152, 153, 154, 155, 156, 157, 158, 159, 160,
              161, 162, 164, 165, 166, 167, 168, 169, 170,
              171, 172, 174, 176, 177, 178, 179, 180, 181]

    # columns to read
    columns_cosmo = ['omega_b', 'omega_cdm', 'sigma8_m', 'n_s', 'alpha_s', 'N_ur', 'w0_fld', 'wa_fld']
    columns_hod = ['logM_cut', 'logM_1', 'sigma', 'alpha', 'kappa', 's', 'B_cen', 'B_sat']

    # output columns
    columns_csv = ['omega_b', 'omega_cdm', 'sigma8_m', 'n_s', 'nrun', 'N_ur', 'w0_fld', 'wa_fld',
        'logM_cut', 'logM_1', 'logsigma', 'alpha', 'kappa', 's', 'B_cen', 'B_sat']

    for cosmo in cosmos:
        cosmo_dict = read_utils.get_abacus_params(cosmo)
        params_cosmo = [cosmo_dict[column] for column in columns_cosmo]

        df = pd.DataFrame(columns=columns_csv)

        params_dir = Path('./hod_params/baseline/')
        params_fn = params_dir / f'hod_params_baseline_c{cosmo:03}.csv'
        hod_params = np.genfromtxt(params_fn, skip_header=1, delimiter=',')

        for i, hod in enumerate(hods):
            logM_cut = hod_params[i, 0]
            logM1 = hod_params[i, 1]
            logsigma = hod_params[i, 2]
            alpha = hod_params[i, 3]
            kappa = hod_params[i, 4]
            s = hod_params[i, 5]
            B_cen = hod_params[i, 6]
            B_sat = hod_params[i, 7]

            params_hod = [logM1, logM_cut, alpha, logsigma, kappa, s, B_cen, B_sat]
            params = params_cosmo + params_hod
            df.loc[i] = params

        output_dir = '/pscratch/sd/e/epaillas/sunbird/data/parameters/abacus_lightcone/dsl'
        output_fn = Path(output_dir, f'AbacusSummit_c{cosmo:03}.csv')
        df.to_csv(output_fn, sep=',', index=False)
