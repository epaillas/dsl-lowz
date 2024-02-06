from cosmopower import cosmopower_NN
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt

# enable latex
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

logger = logging.getLogger("Test")
parser = argparse.ArgumentParser()

parser.add_argument("--features", type=str, required=True)
parser.add_argument("--params", type=str, required=True)
parser.add_argument("--model", type=str, required=True)
args = parser.parse_args()


cp_nn = cosmopower_NN(restore=True, 
                      restore_filename=args.model,
                      )

params = np.load(args.params, allow_pickle=True).item()
test_params = {}
for name in params.keys():
    test_params[name] = list(np.array(params[name])[:100])

predicted = cp_nn.predictions_np(test_params)
features = np.load(args.features)[:100]

res = (predicted - features) / features

# 68, 95, and 99.7 quantiles
q68 = np.quantile(res, [0.16, 0.84], axis=0)
q95 = np.quantile(res, [0.025, 0.975], axis=0)
q99 = np.quantile(res, [0.005, 0.995], axis=0)

# plot quantiles
fig, ax = plt.subplots()
ax.fill_between(np.linspace(-1, 1, 10), q99[0], q99[1], alpha=0.5, label='99\%', color='C2')
ax.fill_between(np.linspace(-1, 1, 10), q95[0], q95[1], alpha=0.5, label='95\%', color='C1')
ax.fill_between(np.linspace(-1, 1, 10), q68[0], q68[1], alpha=1.0, label='68\%', color='C0')
ax.legend(fontsize=15, loc='upper left')
ax.set_xlabel('bin number', fontsize=15)
ax.set_ylabel(r'$\Delta\gamma_t/\gamma_t$', fontsize=15)
# ax.set_ylabel(r'$\Delta w(\theta)/w(\theta)$', fontsize=15)
ax.set_xlim(-1, 1)
ax.tick_params(labelsize=13, which='both')
plt.tight_layout()
# plt.savefig('fig/emulator_accuracy_wtheta.pdf', bbox_inches='tight')
# plt.savefig('fig/emulator_accuracy_gammat.pdf', bbox_inches='tight')
plt.show()
