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


# plotting parameters
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Charter"
plt.rcParams["text.usetex"] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage[bitstream-charter]{mathdesign} \usepackage{amsmath}'
FONTSIZE=16


# define observables
observables = []
observables.append({
            "tex_label": r"$p_{T, \mu_1}$",
            "bins": torch.linspace(20, 150, 50 + 1),
            "yscale": "linear",
        })
observables.append({
            "tex_label": r"$\eta_{\mu_1}$",
            "bins": torch.linspace(-2.5, 2.5, 50 + 1),
            "yscale": "linear"
        })
observables.append({
            "tex_label": r"$\phi_{\mu_1}$",
            "bins": torch.linspace(-3.5, 3.5, 50 + 1),
            "yscale": "linear"
        })
observables.append({
            "tex_label": r"$p_{T, \mu_2}$",
            "bins": torch.linspace(20, 100, 50 + 1),
            "yscale": "linear"
        })
observables.append({
            "tex_label": r"$\eta_{\mu_2}$",
            "bins": torch.linspace(-2.5, 2.5, 50 + 1),
            "yscale": "linear"
        })
observables.append({
            "tex_label": r"$\phi_{\mu_2}$",
            "bins": torch.linspace(-3.5, 3.5, 50 + 1),
            "yscale": "linear"
        })
observables.append({
            "tex_label": r"$p_{T, j_1}$",
            "bins": torch.linspace(20, 200, 50 + 1),
            "yscale": "linear"
        })
observables.append({
            "tex_label": r"$\eta_{j_1}$",
            "bins": torch.linspace(-2.5, 2.5, 50 + 1),
            "yscale": "linear"
        })
observables.append({
            "tex_label": r"$\phi_{j_1}$",
            "bins": torch.linspace(-3.5, 3.5, 50 + 1),
            "yscale": "linear"
        })
observables.append({
            "tex_label": r"$m_{j_1}$",
            "bins": torch.linspace(0, 30, 50 + 1),
            "yscale": "linear"
        })
observables.append({
            "tex_label": r"$N_{j_1}$",
            "bins": torch.arange(0.5, 50.5),
            "yscale": "linear"
        })
observables.append({
            "tex_label": r"$\tau_{1, j_1}$",
            "bins": torch.linspace(-0.1, 0.8, 50 + 1),
            "yscale": "linear"
        })
observables.append({
            "tex_label": r"$\tau_{2, j_1}$",
            "bins": torch.linspace(-0.1, 0.5, 50 + 1),
            "yscale": "linear"
        })
observables.append({
            "tex_label": r"$\tau_{3, j_1}$",
            "bins": torch.linspace(-0.1, 0.4, 50 + 1),
            "yscale": "linear"
        })
observables.append({
            "tex_label": r"$p_{T, j_2}$",
            "bins": torch.linspace(10, 150, 50 + 1),
            "yscale": "linear"
        })
observables.append({
            "tex_label": r"$\eta_{j_2}$",
            "bins": torch.linspace(-2.5, 2.5, 50 + 1),
            "yscale": "linear"
        })
observables.append({
            "tex_label": r"$\phi_{j_2}$",
            "bins": torch.linspace(-3.5, 3.5, 50 + 1),
            "yscale": "linear"
        })
observables.append({
            "tex_label": r"$m_{j_2}$",
            "bins": torch.linspace(0, 30, 50 + 1),
            "yscale": "linear"
        })
observables.append({
            "tex_label": r"$N_{j_2}$",
            "bins": torch.arange(0.5, 40.5),
            "yscale": "linear"
        })
observables.append({
            "tex_label": r"$\tau_{1, j_2}$",
            "bins": torch.linspace(-0.1, 0.8, 50 + 1),
            "yscale": "linear"
        })
observables.append({
            "tex_label": r"$\tau_{2, j_2}$",
            "bins": torch.linspace(-0.1, 0.5, 50 + 1),
            "yscale": "linear"
        })
observables.append({
            "tex_label": r"$\tau_{3, j_2}$",
            "bins": torch.linspace(-0.1, 0.4, 50 + 1),
            "yscale": "linear"
        })

# define correlation functions
def calculate_dimuon_pt(dataset):

    muon1_pt = dataset[:, 0]
    muon1_eta = dataset[:, 1]
    muon1_phi = dataset[:, 2]
    muon2_pt = dataset[:, 3]
    muon2_eta = dataset[:, 4]
    muon2_phi = dataset[:, 5]
    dimuon_pt_2 = muon1_pt ** 2 + muon2_pt ** 2 + 2 * muon1_pt * muon2_pt * np.cos(muon1_phi - muon2_phi)
    return np.sqrt(dimuon_pt_2)

def calculate_dimuon_mass(dataset):

    muon1_pt = dataset[:, 0]
    muon1_eta = dataset[:, 1]
    muon1_phi = dataset[:, 2]
    muon2_pt = dataset[:, 3]
    muon2_eta = dataset[:, 4]
    muon2_phi = dataset[:, 5]
    dimuon_mass_2 = 2 * muon1_pt * muon2_pt * ((np.cosh(muon1_eta - muon2_eta) - np.cos(muon1_phi - muon2_phi)))
    return np.sqrt(dimuon_mass_2)

def calculate_jet_seperation(dataset):

    jet1_eta = dataset[:, 7]
    jet1_phi = dataset[:, 8]
    jet2_eta = dataset[:, 15]
    jet2_phi = dataset[:, 16]
    dR_2 = (jet1_eta - jet2_eta)**2 + (jet1_phi - jet2_phi)**2
    return np.sqrt(dR_2)

# build gen and rec containers
print("Building gen and rec containers")

gen_container = {
"label": "Gen",
"color": "black",
"samples": np.load("/remote/gpu07/huetsch/Z_2j_Gen.npy")[1500000:],
}

rec_container = {
"label": "Rec",
"color": "#0343DE",
"samples": np.load("/remote/gpu07/huetsch/Z_2j_Sim.npy")[1500000:],
}

# change containers to now include all bayesian samples
def make_container(path, label, color):
    container = {
    "label": label,
    "color": color,
    "samples": torch.load(path, map_location=torch.device('cpu'), weights_only=True).numpy()
    }
    return container

# build model containers
print("Building model containers")
path_cfm = "/remote/gpu07/huetsch/Bridges/results/20241022_180947_Z2j_CFM_Bayesian_Transformer_500e/samples.pt"
path_didi = "/remote/gpu07/huetsch/Bridges/results/20241022_180647_Z2j_Didi_0_Bayesian_Transformer_500et"
path_didi = path_cfm
path_didiCond = "/remote/gpu07/huetsch/Bridges/results/20241022_184746_Z2j_DidiCond_1e1_Bayesian_Transformer_500e/samples.pt"

container_cfm = make_container(path_cfm, "CFM", "#A52A2A")
container_didi = make_container(path_didi, "Didi", "#008000")
container_didiCond = make_container(path_didiCond, "Cond. Didi", "#FFA500")


def marginal_plots(path, rec_container, gen_container, *model_containers):
    dims = len(observables)
    with PdfPages(path) as pp:
        for dim in range(dims):
            bins = observables[dim]["bins"]
            hist_rec, _ = np.histogram(rec_container["samples"][:, dim], density=True, bins=bins)
            hist_gen, _ = np.histogram(gen_container["samples"][:, dim], density=True, bins=bins)

            n_bayesian_samples = model_containers[0]["samples"].shape[0]

            model_histograms = []
            for container in model_containers:
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


        correlation_names = ["dimuon_pt", "dimuon_mass", "jet_seperation"]
        correlation_texlabels = [r"$p_{T, \mu \mu }$", r"$m_{\mu \mu }$", r"$\Delta R_{j_1,j_2}$"]
        correlation_functions = [calculate_dimuon_pt, calculate_dimuon_mass, calculate_jet_seperation]
        correlation_units = ["GeV", "GeV", ""]
        correlation_yscales = ["linear", "linear", "linear"]
        correlations_bins = [torch.linspace(0, 200, 50+1), torch.linspace(81, 101, 50+1), torch.linspace(0, 8, 50+1)]

        for i in range(len(correlation_names)):
            bins = correlations_bins[i]
            hist_rec, _ = np.histogram(correlation_functions[i](rec_container["samples"]), density=True, bins=bins)
            hist_gen, _ = np.histogram(correlation_functions[i](gen_container["samples"]), density=True, bins=bins)

            n_bayesian_samples = model_containers[0]["samples"].shape[0]
            model_histograms = []
            for container in model_containers:  
                hist_unfolded = np.stack([np.histogram(correlation_functions[i](container["samples"][sample, :, :]), density=True, bins=bins)[0] for sample in range(n_bayesian_samples)])
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


            axs[0].set_yscale(correlation_yscales[i])
            axs[0].legend(frameon=False, fontsize=FONTSIZE)
            axs[0].set_ylabel("Normalized", fontsize=FONTSIZE)
            axs[0].tick_params(axis="both", labelsize=FONTSIZE)

            for i, hist_unfolded in enumerate(model_histograms):
                axs[1].step(bins[1:], hist_unfolded[-1] / hist_gen, color=model_containers[i]["color"], where="post")
                axs[1].fill_between(bins[1:], (hist_unfolded.mean(axis=0)-hist_unfolded.std(axis=0)) / hist_gen, (hist_unfolded.mean(axis=0)+hist_unfolded.std(axis=0)) / hist_gen, alpha=0.2, color=model_containers[i]["color"], step="post")
                axs[1].fill_between(bins[1:], (hist_unfolded[-1]-hist_unfolded.std(axis=0)) / hist_gen, (hist_unfolded[-1]+hist_unfolded.std(axis=0)) / hist_gen, alpha=0.2, color=model_containers[i]["color"], step="post")

            axs[1].set_ylabel(r"$\frac{\mathrm{Model}}{\mathrm{True}}$", fontsize=FONTSIZE)

            axs[1].set_yticks([0.9, 1, 1.1])
            axs[1].set_ylim([0.81, 1.19])
            axs[1].axhline(y=1., c="black", ls="--", lw=0.7)
            axs[1].axhline(y=1.2, c="black", ls="dotted", lw=0.5)
            axs[1].axhline(y=0.8, c="black", ls="dotted", lw=0.5)
            axs[1].tick_params(axis="both", labelsize=FONTSIZE)

            plt.xlabel(correlation_texlabels[i], fontsize=FONTSIZE)
            plt.savefig(pp, format="pdf", bbox_inches="tight", pad_inches=0.05)
            plt.close()

print("Plotting marginal plots")
marginal_plots("paperplots/marginal_plots_Z2j.pdf", rec_container, gen_container, container_cfm, container_didi, container_didiCond)
