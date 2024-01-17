import argparse
import yaml
import time
from pathlib import Path
from sunbird.inference import HMC, Nested

if __name__ == "__main__":
    output_path = Path("/pscratch/sd/e/epaillas/sunbird/chains/dsl/")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, default="infer_abacus.yaml"
    )
    # Make sure it reads from dataset with fixed hod
    parser.add_argument("--cosmology", type=int, default=0)
    parser.add_argument("--suffix", type=str, default=None)
    args = parser.parse_args()
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    config["data"]["observation"]["get_obs_args"]["cosmology"] = args.cosmology
    loss = config["theory_model"]["args"]["loss"]
    # smin = config["slice_filters"]["s"][0]
    # smax = config["slice_filters"]["s"][1]
    statistics = "_".join([i for i in config["statistics"]])
    # multipoles = ''.join([str(i) for i in config["select_filters"]["multipoles"]])
    dir_store = f"abacus_c{args.cosmology}_{statistics}"
    if args.suffix is not None:
        dir_store += f"_{args.suffix}"
    config["inference"]["output_dir"] = output_path / dir_store
    print("output dir")
    print(config["inference"]["output_dir"])
    nested = Nested.from_config_dict(
        config=config,
    )
    t0 = time.time()
    print(f"Fitting parameters {nested.param_names}")
    nested()
    print("Fitting took = ", time.time() - t0)
