from getdist import plots, MCSamples
from getdist.mcsamples import loadMCSamples
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sunbird.cosmology.growth_rate import Growth
from cosmoprimo.fiducial import AbacusSummit
import argparse

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

labels = {
    "omega_b": r"\omega_{\rm b}",
    "omega_cdm": r"\omega_{\rm cdm}",
    "sigma8_m": r"\sigma_8",
    "n_s": r"n_s",
    "nrun": r"\alpha_s",
    "N_ur": r"N_{\rm ur}",
    "w0_fld": r"w_0",
    "wa_fld": r"w_a",
    "logM_1": "logM_1",
    "logM_cut": r"logM_{\rm cut}",
    "alpha": r"\alpha",
    "alpha_s": r"\alpha_{\rm vel, s}",
    "alpha_c": r"\alpha_{\rm vel, c}",
    "logsigma": r"\log \sigma",
    "kappa": r"\kappa",
    "s": r"s",
    "B_cen": r"B_{\rm cen}",
    "B_sat": r"B_{\rm sat}",
    "fsigma8": r"f \sigma_8",
    "Omega_m": r"\Omega_{\rm m}",
    "h": r"h",
}


def read_dynesty_chain(filename, add_fsigma8=False, redshift=0.525):
    data = pd.read_csv(filename)
    param_names = list(data.columns[4:])
    if add_fsigma8:
        growth = Growth(
            emulate=True,
        )
        param_names.append("fsigma8")
        data['fsigma8'] = growth.get_fsigma8(
            omega_b = data['omega_b'].to_numpy(),
            omega_cdm = data['omega_cdm'].to_numpy(),
            sigma8 = data['sigma8_m'].to_numpy(),
            n_s = data['n_s'].to_numpy(),
            N_ur = np.ones_like(data['omega_b'].to_numpy()) * 2.0328,
            w0_fld = np.ones_like(data['omega_b'].to_numpy()) * -1.0,
            wa_fld = np.ones_like(data['omega_b'].to_numpy()) * 0.0,
            z=redshift,
        )
        param_names.append("h")
        data['h'] = growth.get_emulated_h(
            omega_b = data['omega_b'].to_numpy(),
            omega_cdm = data['omega_cdm'].to_numpy(),
            sigma8 = data['sigma8_m'].to_numpy(),
            n_s = data['n_s'].to_numpy(),
            N_ur = np.ones_like(data['omega_b'].to_numpy()) * 2.0328,
            w0_fld = np.ones_like(data['omega_b'].to_numpy()) * -1.0,
            wa_fld = np.ones_like(data['omega_b'].to_numpy()) * 0.0,
        )
        param_names.append("Omega_m")
        data['Omega_m'] = growth.Omega_m0(
            data['omega_cdm'],
            data['omega_b'],
            data['h'],
        )
    data = data.to_numpy()
    chain = data[:, 4:]
    loglikes = data[:, 0] * -1
    weights = np.exp(data[:, 1] - data[-1, 2])
    return param_names, chain, weights, loglikes

def read_hmc_chain(filename, add_fsigma8=False, redshift=0.525,):
    data = pd.read_csv(filename)
    param_names = list(data.columns)
    if add_fsigma8:
        growth = Growth(
            emulate=True,
        )
        param_names.append("fsigma8")
        data['fsigma8'] = growth.get_fsigma8(
            omega_b = data['omega_b'].to_numpy(),
            omega_cdm = data['omega_cdm'].to_numpy(),
            sigma8 = data['sigma8_m'].to_numpy(),
            n_s = data['n_s'].to_numpy(),
            N_ur = np.ones_like(data['omega_b'].to_numpy()) * truth['N_ur'],
            w0_fld = np.ones_like(data['omega_b'].to_numpy()) * truth['w0_fld'],
            wa_fld = np.ones_like(data['omega_b'].to_numpy()) * truth['wa_fld'],
            z=redshift,
        )
        param_names.append("h")
        data['h'] = growth.get_emulated_h(
            omega_b = data['omega_b'].to_numpy(),
            omega_cdm = data['omega_cdm'].to_numpy(),
            sigma8 = data['sigma8_m'].to_numpy(),
            n_s = data['n_s'].to_numpy(),
            N_ur = np.ones_like(data['omega_b'].to_numpy()) * truth['N_ur'],
            w0_fld = np.ones_like(data['omega_b'].to_numpy()) * truth['w0_fld'],
            wa_fld = np.ones_like(data['omega_b'].to_numpy()) * truth['wa_fld'],
        )
        param_names.append("Omega_m")
        data['Omega_m'] = growth.Omega_m0(
            data['omega_cdm'],
            data['omega_b'],
            data['h'],
        )
    data = data.to_numpy()
    return param_names, data, None, None

def get_MCSamples(filename, add_fsigma8=False, redshift=0.525, hmc=True,):
    priors = {
        "omega_b": [0.0207, 0.0243],
        "omega_cdm": [0.1032, 0.140],
        "sigma8_m": [0.678, 0.938],
        "n_s": [0.9012, 1.025],
        "nrun": [-0.038, 0.038],
        "N_ur": [1.188, 2.889],
        "w0_fld": [-1.22, -0.726],
        "wa_fld": [-0.628, 0.621],
        "logM1": [13.2, 14.4],
        "logM_cut": [12.4, 13.3],
        "alpha": [0.7, 1.5],
        "alpha_s": [0.7, 1.3],
        "alpha_c": [0.0, 0.5],
        "logsigma": [-3.0, 0.0],
        "kappa": [0.0, 1.5],
        "s": [-1.0, 1.0],
        "B_cen": [-1.0, 1.0],
        "B_sat": [-1.0, 1.0],
    }
    if hmc:
        names, chain, weights, loglikes = read_hmc_chain(
            filename,
            add_fsigma8=add_fsigma8,
            redshift=redshift,
        )
    else:
        names, chain, weights, loglikes = read_dynesty_chain(
            filename,
            add_fsigma8=add_fsigma8,
            redshift=redshift,
        )

    samples = MCSamples(
        samples=chain,
        weights=weights,
        labels=[labels[n] for n in names],
        names=names,
        ranges=priors,
        loglikes=loglikes
    )
    # print(samples.getTable(limit=1).tableTex())
    return samples, names

def get_true_params(
    cosmology,
    hod_idx,
    add_fsigma8=False,
    redshift=0.5,
):
    param_dict = dict(
        pd.read_csv(
            f"/pscratch/sd/e/epaillas/sunbird/data/parameters/abacus_lightcone/dsl/AbacusSummit_c{str(cosmology).zfill(3)}.csv"
        ).iloc[hod_idx]
    )
    if add_fsigma8:
        cosmo = AbacusSummit(cosmology)
        param_dict["fsigma8"] = cosmo.sigma8_z(redshift) * cosmo.growth_rate(redshift)
    return param_dict


args  = argparse.ArgumentParser()
args.add_argument('--chain_dir', type=str, default='/pscratch/sd/e/epaillas/sunbird/chains/voxel_voids_fullap')
args.add_argument('--param_space', type=str, default='cosmo')
args.add_argument('--cosmology', type=int, default=0)
args = args.parse_args()

chain_dir = Path(args.chain_dir)

smin = 0.7
smax = 150
redshift = 0.525
growth = Growth(emulate=True,)
param_space = 'base_bbn'

chain_handles = [
    f'/pscratch/sd/e/epaillas/sunbird/chains/dsl/abacus_c0_gammat'
]
chain_labels = [
    'gammat_',
]


if args.param_space == 'cosmo':
    params_toplot = [
        'omega_cdm', 'sigma8_m', 'n_s',
        # 'nrun', 'N_ur', 'w0_fld', 'wa_fld',
        # 'logM_cut', 'logM1', 'alpha',
        # 'logsigma', 'kappa',
    ]

truth = get_true_params(args.cosmology, 0, add_fsigma8=False, redshift=redshift)
# truth['omega_ncdm'] = 0.00064420
# if args.cosmology == 0:
#     truth['h'] = 0.6736
# elif args.cosmology == 3:
#     truth['h'] = 0.7160

# truth['Omega_m'] = (truth['omega_cdm'] + truth['omega_b'] + truth['omega_ncdm']) / truth['h'] ** 2

samples_list = []

for i in range(len(chain_handles)):
    chain_fn = chain_dir / chain_handles[i] / 'results.csv'
    print(chain_fn)
    hmc = False
    samples, names = get_MCSamples(chain_fn, hmc=hmc, add_fsigma8=False,)
    samples_list.append(samples)
    print(samples.getLikeStats())
    print(chain_labels[i])
    print(samples.getTable(limit=1).tableTex())

samples_planck_boss = loadMCSamples('/global/cfs/cdirs/desi/users/plemos/planck/chains/base/plikHM_TTTEEE_lowl_lowE_lensing/base_plikHM_TTTEEE_lowl_lowE_lensing', settings={'ignore_rows': 0.3})
samples_planck_boss.addDerived(samples_planck_boss.getParams().omegabh2, name='omega_b', label='\omega_{\rm cdm}')
samples_planck_boss.addDerived(samples_planck_boss.getParams().omegach2, name='omega_cdm', label='\omega_{\rm cdm}')
samples_planck_boss.addDerived(samples_planck_boss.getParams().omegam, name='Omega_m', label='\Omega_{\rm m}')
samples_planck_boss.addDerived(samples_planck_boss.getParams().sigma8, name='sigma8_m', label='\sigma_8')
samples_planck_boss.addDerived(samples_planck_boss.getParams().ns, name='n_s', label='n_s')
# samples_list.append(samples_planck_boss)

param_limits = {
    'omega_cdm': [0.112, 0.1293],
    # 'sigma8_m': [0.68, 0.92],
    # 'n_s': [0.9012, 1.0],
    # 'logsigma': [-0.7, -0.3],
    # 'alpha_s': [0.7, 1.0],
}

colors = ['lightcoral', 'royalblue', 'orange']
bright = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377', '#BBBBBB']
retro = ['#4165c0', '#e770a2', '#5ac3be', '#696969', '#f79a1e', '#ba7dcd']

g = plots.get_subplot_plotter(width_inch=8)
g.settings.axis_marker_lw = 1.0
g.settings.axis_marker_ls = '-'
g.settings.title_limit_labels = False
g.settings.axis_marker_color = 'k'
g.settings.legend_colored_text = True
g.settings.figure_legend_frame = False
g.settings.linewidth = 2.0
g.settings.linewidth_contour = 3.0
g.settings.legend_fontsize = 22
g.settings.axes_fontsize = 17
g.settings.axes_labelsize = 22
g.settings.axis_tick_x_rotation = 45
g.settings.axis_tick_max_labels = 6
g.settings.solid_colors = retro

g.triangle_plot(
    roots=samples_list,
    params=params_toplot,
    filled=True,
    legend_labels=chain_labels,
    legend_loc='upper right',
    title_limit=1,
    # param_limits=param_limits,
    markers=truth,
)
# plt.show()
# output_fn = f'{args.param_space}_inference_abacus_voxel_voids_c0.png'
# plt.savefig(output_fn, bbox_inches='tight', dpi=300)
plt.show()