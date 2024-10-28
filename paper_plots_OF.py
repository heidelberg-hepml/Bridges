import math
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import time
import matplotlib
import h5py
from matplotlib.colors import LogNorm


# plotting parameters
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Charter"
plt.rcParams["text.usetex"] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage[bitstream-charter]{mathdesign} \usepackage{amsmath}'
FONTSIZE=16


# define observables
observables = []
observables.append({
            "tex_label": r"\text{Jet mass } m",
            "bins": torch.linspace(3, 60, 50 + 1),
            "yscale": "linear"
        })
observables.append({
            "tex_label": r"\text{Jet multiplicity } N",
            "bins": torch.arange(5.5, 60.5),
            "yscale": "linear"
        })
observables.append({
            "tex_label": r"$\text{N-subjettiness ratio } \tau_{21}$",
            "bins": torch.linspace(0.2, 1.1, 50 + 1),
            "yscale": "linear"
        })
observables.append({
            "tex_label": r"\text{Jet width } w",
            "bins": torch.linspace(0.05, 0.55, 50 + 1),
            "yscale": "log"
        })
observables.append({
            "tex_label": r"$\text{Groomed mass }\log \rho$",
            "bins": torch.linspace(-12, -2.5, 50 + 1),
            "yscale": "linear"
        })
observables.append({
            "tex_label": r"$\text{Groomed momentum fraction }z_g$",
            "bins": torch.linspace(0.05, 0.55, 50 + 1),
            "yscale": "log"
        })


# define migration plots
def migration_plots(path, rec_container, gen_container, *model_containers):
    
    # number of models. hardcoded for now
    n_models = len(model_containers)
    
    fig, axs = plt.subplots(n_models+1, 3, figsize=(10,14))
    
    # observables to plot. hardcoded for now
    obsverbales_to_plot = [1,2,4]
    labels = ["m", "N", r"$\tau_{21}$", "w", r"$\log \rho$", r"$z_g$"]
    
    # plot truth migration
    for i_plot, i in enumerate(obsverbales_to_plot):
        o = observables[i]
        o_label = labels[i]
        
        bins = o["bins"]
        gen = gen_container["samples"][:, i]
        rec = rec_container["samples"][:, i]
        
        axs[0, i_plot].hist2d(gen, rec, density=True, bins=bins, rasterized=True, norm=LogNorm())
        axs[0, i_plot].set_title("Truth", fontsize=FONTSIZE)
        axs[0, i_plot].set_xlabel(o_label+"  (Rec)", fontsize=FONTSIZE)
        axs[0, i_plot].set_ylabel(o_label+"  (Gen)", fontsize=FONTSIZE)

        
    # plot model migrations
    for n_model in range(len(model_containers)):
        model = model_containers[n_model]
        
        label = ["CFM", "DM", "CDM", "SB-SC", "SB-uncond"][n_model]
        
        for i_plot, i in enumerate(obsverbales_to_plot):
            o = observables[i]
            o_label = labels[i]
            bins = o["bins"]
            unfold = model["samples"][-1, :, i]
            rec = rec_container["samples"][:, i]
            axs[n_model+1, i_plot].hist2d(unfold, rec, density=True, bins=bins, rasterized=True, norm=LogNorm())
            axs[n_model+1, i_plot].set_title(label, fontsize=FONTSIZE)
            axs[n_model+1, i_plot].set_xlabel(o_label+"  (Rec)", fontsize=FONTSIZE)
            axs[n_model+1, i_plot].set_ylabel(o_label+"  (Gen)", fontsize=FONTSIZE)

    plt.tight_layout()
    plt.savefig(path, format="pdf", bbox_inches="tight")
    plt.close()
    

# helper function to make model container
def make_container(path, label, color):
    container = {
    "label": label,
    "color": color,
    "samples": torch.load(path, map_location=torch.device('cpu'), weights_only=True).numpy()
    }
    return container

# build gen and rec containers
print("Building gen and rec containers")
path_testdata = "/remote/gpu07/huetsch/data/omnifold_data/OmniFold_big/OmniFold_test.h5"
with h5py.File(path_testdata, "r") as f:
    gen_container = {
    "label": "Gen",
    "color": "black",
    "samples": np.array(f["hard"])[:4000000, [0, 2, 5, 1, 3, 4]],
    }

    rec_container = {
    "label": "Rec",
    "color": "#0343DE",
    "samples": np.array(f["reco"])[:4000000, [0, 2, 5, 1, 3, 4]],
    }

# build model containers
print("Building model containers")
path_cfm = "/remote/gpu07/huetsch/Bridges/results/20241022_213959_OF6_CFM_Bayesian_100e/samples.pt"
path_didi = "/remote/gpu07/huetsch/Bridges/results/20241022_214044_OF6_Didi_0e_Bayesian_100e/samples.pt"
path_didiCond = "/remote/gpu07/huetsch/Bridges/results/20241022_214126_OF6_DidiCond_1e_Bayesian_100e/samples.pt"

container_cfm = make_container(path_cfm, "CFM", "#A52A2A")
container_didi = make_container(path_didi, "Didi", "#008000")
container_didiCond = make_container(path_didiCond, "Cond. Didi", "#FFA500")

path_SB_SC = "/remote/gpu07/huetsch/Bridges/SB_results_OF/SBUnfold_OmniFolddata_largeset_SC_unfolded.npy"
path_SB_uncond = "/remote/gpu07/huetsch/Bridges/SB_results_OF/SBUnfold_OmniFolddata_largeset_unfolded.npy"

container_SB_SC = {
    "label": "SB-SC",
    "color": "brown",
    "samples": np.load(path_SB_SC, allow_pickle=True)[:, :4000000, [0, 2, 5, 1, 3, 4]]
}
container_SB_uncond = {
    "label": "SB-uncond",
    "color": "grey",
    "samples": np.load(path_SB_uncond, allow_pickle=True)[:, :4000000, [0, 2, 5, 1, 3, 4]]
}

# plot migration plots
print("Plotting migration plots")
migration_plots("paperplots/migration_plots_OF.pdf", rec_container, gen_container, container_cfm, container_didi, container_didiCond, container_SB_SC, container_SB_uncond)

def marginal_plots(path, rec_container, gen_container, *model_containers):
    dims = len(observables)
    with PdfPages(path) as pp:
        for dim in range(dims):
            bins = observables[dim]["bins"]
            hist_rec, _ = np.histogram(rec_container["samples"][:, dim], density=True, bins=bins)
            hist_gen, _ = np.histogram(gen_container["samples"][:, dim], density=True, bins=bins)

            model_histograms = []
            for container in model_containers:
                n_bayesian_samples = container["samples"].shape[0]
                hist_unfolded = np.stack([np.histogram(container["samples"][sample, :, dim], density=True, bins=bins)[0] for sample in range(n_bayesian_samples)])
                model_histograms.append(hist_unfolded)

            fig1, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.00})
            fig1.tight_layout(pad=0.6, w_pad=0.5, h_pad=0.6, rect=(0.07, 0.06, 0.99, 0.95))

            # histogram
            axs[0].step(bins[1:], hist_rec, label="Rec", linewidth=1.0, where="post", color="blue")
            axs[0].step(bins[1:], hist_gen, label="Gen", linewidth=1.0, where="post", color="black")
            for i, hist_unfolded in enumerate(model_histograms):
                axs[0].step(bins[1:], hist_unfolded[-1], label=model_containers[i]["label"], linewidth=1.0, where="post", color=model_containers[i]["color"])
                axs[0].fill_between(bins[1:], hist_unfolded.mean(axis=0) - hist_unfolded.std(axis=0), hist_unfolded.mean(axis=0) + hist_unfolded.std(axis=0), alpha=0.2, color=model_containers[i]["color"], step="post")
                axs[0].fill_between(bins[1:], hist_unfolded[-1] - hist_unfolded.std(axis=0), hist_unfolded[-1] + hist_unfolded.std(axis=0), alpha=0.2, color=model_containers[i]["color"], step="post")

            axs[0].set_yscale(observables[dim]["yscale"])
            axs[0].legend(frameon=False, fontsize=FONTSIZE)
            axs[0].set_ylabel("Normalized", fontsize=FONTSIZE)
            axs[0].tick_params(axis="both", labelsize=FONTSIZE)

            # ratio panel   
            for i, hist_unfolded in enumerate(model_histograms):
                axs[1].step(bins[1:], hist_unfolded[-1] / hist_gen, color=model_containers[i]["color"], where="post")
                axs[1].fill_between(bins[1:], (hist_unfolded.mean(axis=0)-hist_unfolded.std(axis=0)) / hist_gen, (hist_unfolded.mean(axis=0)+hist_unfolded.std(axis=0)) / hist_gen, alpha=0.2, color=model_containers[i]["color"], step="post")
                axs[1].fill_between(bins[1:], (hist_unfolded[-1]-hist_unfolded.std(axis=0)) / hist_gen, (hist_unfolded[-1]+hist_unfolded.std(axis=0)) / hist_gen, alpha=0.2, color=model_containers[i]["color"], step="post")

            axs[1].set_ylabel(r"$\frac{\mathrm{Model}}{\mathrm{True}}$", fontsize=FONTSIZE)
            axs[1].set_yticks([0.9,1,1.1])
            axs[1].set_ylim([0.81, 1.19])
            axs[1].axhline(y=1., c="black", ls="--", lw=0.7)
            axs[1].axhline(y=1.2, c="black", ls="dotted", lw=0.5)
            axs[1].axhline(y=0.8, c="black", ls="dotted", lw=0.5)
            axs[1].tick_params(axis="both", labelsize=FONTSIZE)

            plt.xlabel(observables[dim]["tex_label"], fontsize=FONTSIZE)
            plt.savefig(pp, format="pdf", bbox_inches="tight", pad_inches=0.05)
            plt.close()

print("Plotting marginal plots")
marginal_plots("paperplots/marginal_plots_OF.pdf", rec_container, gen_container, container_cfm, container_didi, container_didiCond, container_SB_SC, container_SB_uncond)
