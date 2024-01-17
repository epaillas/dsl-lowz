from __future__ import print_function
import numpy as np
import random 
import multiprocessing
from cosmopower_NN import cosmopower_NN
import tensorflow as tf
import argparse
import logging


logger = logging.getLogger("Train")
parser = argparse.ArgumentParser()
parser.add_argument("--params", type=str, required=True)
parser.add_argument("--features", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--n_nodes", type=int, default=512)
parser.add_argument("--n_layers", type=int, default=4)
parser.add_argument("--device", type=str, choices=["cpu", "gpu"], default="gpu")
args = parser.parse_args()


params = np.load(args.params, allow_pickle=True).item()
train_params = {}
for name in params.keys():
    train_params[name]=list(np.array(params[name])[100:])
features = np.load(args.features)[100:]

# set up the neural network
cp_nn = cosmopower_NN(parameters=list(params.keys()), 
                    modes=np.linspace(-1,1,features.shape[1]), 
                    n_hidden=[args.n_nodes for i in range(args.n_layers)],
                    verbose=True,
                    )
with tf.device(args.device):
    # train
    cp_nn.train(training_parameters=train_params,
                training_features=features,
                filename_saved_model=args.output,
                # cooling schedule
                validation_split=0.1,
                learning_rates=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
                batch_sizes=[100, 100, 100, 100, 100],
                gradient_accumulation_steps = [1, 1, 1, 1, 1],
                # early stopping set up
                patience_values = [100,100,100,100,100],
                max_epochs = [1000,1000,1000,1000,1000],
                )