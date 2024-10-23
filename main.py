import argparse
import yaml
import os
import torch
import builtins
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from omnifold_dataset import Omnifold
from two_gaussians_dataset import TwoGaussians
from Z_2j_dataset import Z_2j_dataset
from models import CFM, Didi
from plots import marginal_plots, migration_plots, single_event_plots, loss_plot, time_evolution_plots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('type')
    parser.add_argument('path')
    args = parser.parse_args()

    if args.type == "train":
        # read in the parameters
        with open(args.path, 'r') as f:
            params = yaml.safe_load(f)

        # create a results dir and save parameters to it
        dir_path = os.path.dirname(os.path.realpath(__file__))
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + params["run_name"]
        run_dir = os.path.join(dir_path, "results", run_name)
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, "params.yaml"), 'w') as f:
            yaml.dump(params, f)

        # create log file
        log_file = os.path.join(run_dir, "train_log.txt")
        with open(log_file, 'w') as f:
            f.write(f"Starting run {run_name}\n")

    elif args.type == "plot":
        # read in saved run directory and parameters
        run_dir = args.path
        with open(os.path.join(run_dir, "params.yaml"), 'r') as f:
            params = yaml.safe_load(f)

        log_file = os.path.join(run_dir, "plot_log.txt")
        with open(log_file, 'w') as f:
            f.write(f"Starting plotting for run from dir {run_dir}\n")

    else:
        raise NotImplementedError(f"type {args.type} not recognised")
    
    # Redefine the print function to also write to the log file
    original_print = print
    def custom_print(*args, **kwargs):
        original_print(*args, **kwargs)
        with open(log_file, 'a') as f:
            original_print(*args, file=f, **kwargs)
    builtins.print = custom_print

    # Redirect error messages to the log file
    # There is no 'error' attribute in builtins, so we'll use sys.stderr instead
    import sys
    #original_stderr = sys.stderr
    #def custom_error(*args, **kwargs):
    #    original_stderr.write(*args, **kwargs)
    #    with open(log_file, 'a') as f:
    #        original_stderr.write(*args, file=f, **kwargs)
    #sys.stderr.write = custom_error

    # look for GPUs
    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"
    print(f"Using device {device}")

    # get the dataset
    print("Building dataset", params["dataset_params"]["type"])
    dataset = eval(params["dataset_params"]["type"])(params["dataset_params"])
    # get gen and rec dimension from the dataset
    dims_gen = dataset.gen.shape[-1]
    dims_rec = dataset.rec.shape[-1]

    # build the model
    print("Building model", params["model_params"]["type"])
    model = eval(params["model_params"]["type"])(dims_gen, dims_rec, params["model_params"]).to(device)

    if args.type == "train":
        # train the model
        print("Training model")
        model.train(dataset.gen, dataset.rec, None)

        # Save the model
        print("Saving model")
        os.makedirs(os.path.join(run_dir, "model"), exist_ok=True)
        model_path = os.path.join(run_dir, "model", f"model.pth")
        torch.save(model.state_dict(), model_path)

        # build testset
        print("Loading testset")
        dataset.init_dataset(test=True)
        dataset.apply_preprocessing()

        # evaluate the model
        print("Sampling model")
        dataset.unfolded = model.evaluate(dataset.rec)
        print("Sampling single event unfoldings")
        dataset.single_event_unfolded = model.single_event_unfolding(dataset.rec[:5])
        dataset.apply_preprocessing(reverse=True)

        # save samples
        print("Saving samples")
        file_samples = os.path.join(run_dir, f"samples.pt")
        torch.save(dataset.unfolded.detach().cpu(), file_samples)
        file_singleevents = os.path.join(run_dir, f"single_event_unfoldings.pt")
        torch.save(dataset.single_event_unfolded.detach().cpu(), file_singleevents)

    elif args.type == "plot":
        # Load saved model
        print("Loading model")
        model.load_state_dict(torch.load(os.path.join(run_dir, "model", f"model.pth"),  map_location=device, weights_only=True))

        # try to load already generated samples
        file_samples = os.path.join(run_dir, f"samples.pt")
        file_singleevents = os.path.join(run_dir, f"single_event_unfoldings.pt")
        try:
            print("Loading already generated samples")
            samples = torch.load(file_samples, map_location="cpu", weights_only=True)
            single_event_samples = torch.load(file_singleevents, map_location="cpu", weights_only=True)
            
            # build testset
            print("Loading testset")
            dataset.init_dataset(test=True)
            dataset.unfolded = samples
            dataset.single_event_unfolded = single_event_samples
        except:
            print("Failed to load already generated samples")

            # build testset
            print("Loading testset")
            dataset.init_dataset(test=True)
            dataset.apply_preprocessing()

            # evaluate the model
            print("Sampling model")
            dataset.unfolded = model.evaluate(dataset.rec)
            print("Sampling single event unfoldings")
            dataset.single_event_unfolded = model.single_event_unfolding(dataset.rec[:5])
            dataset.apply_preprocessing(reverse=True)

            print("Saving samples")
            file_samples = os.path.join(run_dir, f"samples.pt")
            torch.save(dataset.unfolded.detach().cpu(), file_samples)
            file_singleevents = os.path.join(run_dir, f"single_event_unfoldings.pt")
            torch.save(dataset.single_event_unfolded.detach().cpu(), file_singleevents)

    # make plots
    print("Making plots")
    if hasattr(model, "trainlosses"):
        file_lossplots = os.path.join(run_dir, f"plots_loss.pdf")
        loss_plot(file_lossplots, model.trainlosses, model.vallosses)
    file_marginalplots = os.path.join(run_dir, f"plots_marginals.pdf")
    marginal_plots(file_marginalplots, dataset)
    #file_migrationplots = os.path.join(run_dir, f"plots_migration.pdf")
    #migration_plots(file_migrationplots, dataset)
    #file_singleeventplots = os.path.join(run_dir, f"plots_singleevent.pdf")
    #single_event_plots(file_singleeventplots, dataset)
    #file_timeevolutionplots = os.path.join(run_dir, f"plots_evolution.pdf")
    #time_evolution_plots(file_timeevolutionplots, dataset)

    print("Done")
if __name__ == '__main__':
    main()
