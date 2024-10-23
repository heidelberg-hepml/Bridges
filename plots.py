import numpy as np
import torch
import matplotlib.pyplot as plt
from Z_2j_dataset import Z_2j_dataset
from matplotlib.backends.backend_pdf import PdfPages


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Charter"
plt.rcParams["text.usetex"] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage[bitstream-charter]{mathdesign} \usepackage{amsmath}'
FONTSIZE=16


def marginal_plots(path, dataset, weights=None):
    dims = len(dataset.observables)
    with PdfPages(path) as pp:
        for dim in range(dims):
            bins = dataset.observables[dim]["bins"]
            hist_rec, _ = np.histogram(dataset.rec[:, dim].cpu(), density=True, bins=bins)
            hist_gen, _ = np.histogram(dataset.gen[:, dim].cpu(), density=True, bins=bins)

            if len(dataset.unfolded.shape) == 2:
                hist_unfolded, _ = np.histogram(dataset.unfolded[:, dim].cpu(), density=True, bins=bins).unsqueeze(0)
            else:
                n_bayesian_samples = dataset.unfolded.shape[0]
                hist_unfolded = np.stack([np.histogram(dataset.unfolded[sample, :, dim].cpu(), density=True, bins=bins)[0] for sample in range(n_bayesian_samples)])
            if weights is not None:
                hist_unfolded_reweighted, _ = np.histogram(dataset.unfolded[:, dim].cpu(), weights=weights, density=True, bins=bins)

            fig1, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.00})
            fig1.tight_layout(pad=0.6, w_pad=0.5, h_pad=0.6, rect=(0.07, 0.06, 0.99, 0.95))

            # histogram
            axs[0].step(bins[1:], hist_rec, label="Rec", linewidth=1.0, where="post", color="blue")
            axs[0].step(bins[1:], hist_gen, label="Gen", linewidth=1.0, where="post", color="black")
            axs[0].step(bins[1:], hist_unfolded[-1], label="Unfolded", linewidth=1.0, where="post", color="red")
            if n_bayesian_samples > 1:
                axs[0].step(bins[1:], hist_unfolded.mean(axis=0), label="Unfolded Mean", linewidth=1.0, where="post", color="green")

            if n_bayesian_samples > 1:
                # include std of bayesian histograms as error band for mean and unfolded
                #axs[0].step(bins[1:], hist_unfolded.mean(axis=0) - hist_unfolded.std(axis=0), alpha=0.2, color="green")
                #axs[0].step(bins[1:], hist_unfolded.mean(axis=0) + hist_unfolded.std(axis=0), alpha=0.2, color="green")
                axs[0].fill_between(bins[1:], hist_unfolded.mean(axis=0) - hist_unfolded.std(axis=0), hist_unfolded.mean(axis=0) + hist_unfolded.std(axis=0), alpha=0.2, color="green", step="post")
                #axs[0].step(bins[1:], hist_unfolded[-1] - hist_unfolded.std(axis=0), alpha=0.2, color="red")
                #axs[0].step(bins[1:], hist_unfolded[-1] + hist_unfolded.std(axis=0), alpha=0.2, color="red")
                axs[0].fill_between(bins[1:], hist_unfolded[-1] - hist_unfolded.std(axis=0), hist_unfolded[-1] + hist_unfolded.std(axis=0), alpha=0.2, color="red", step="post")

            if weights is not None:
                axs[0].step(bins[1:], hist_unfolded_reweighted, label="Unfolded (reweighted)", linewidth=1.0, where="post", color="green")
            axs[0].set_yscale(dataset.observables[dim]["yscale"])

            axs[0].legend(frameon=False, fontsize=FONTSIZE)
            axs[0].set_ylabel("Normalized", fontsize=FONTSIZE)
            axs[0].tick_params(axis="both", labelsize=FONTSIZE)

            # ratio panel
            axs[1].step(bins[1:], hist_unfolded[-1] / hist_gen, color="red", where="post")
            if n_bayesian_samples > 1:
                axs[1].step(bins[1:], hist_unfolded.mean(axis=0) / hist_gen, color="green", where="post")
            if weights is not None:
                axs[1].step(bins[1:], hist_unfolded_reweighted / hist_gen, color="green", where="post")

            if n_bayesian_samples > 1:
                #axs[1].step(bins[1:], hist_unfolded.mean(axis=0) / hist_gen - hist_unfolded.std(axis=0) / hist_gen, alpha=0.2, color="green")
                #axs[1].step(bins[1:], hist_unfolded.mean(axis=0) / hist_gen + hist_unfolded.std(axis=0) / hist_gen, alpha=0.2, color="green")
                axs[1].fill_between(bins[1:], (hist_unfolded.mean(axis=0)-hist_unfolded.std(axis=0)) / hist_gen, (hist_unfolded.mean(axis=0)+hist_unfolded.std(axis=0)) / hist_gen, alpha=0.2, color="green", step="post")
                #axs[1].step(bins[1:], hist_unfolded[-1] / hist_gen - hist_unfolded.std(axis=0) / hist_gen, alpha=0.2, color="red")
                #axs[1].step(bins[1:], hist_unfolded[-1] / hist_gen + hist_unfolded.std(axis=0) / hist_gen, alpha=0.2, color="red")
                axs[1].fill_between(bins[1:], (hist_unfolded[-1]-hist_unfolded.std(axis=0)) / hist_gen, (hist_unfolded[-1]+hist_unfolded.std(axis=0)) / hist_gen, alpha=0.2, color="red", step="post")

            axs[1].set_ylabel(r"$\frac{\mathrm{Model}}{\mathrm{True}}$", fontsize=FONTSIZE)
            axs[1].set_yticks([0.9,1,1.1])
            axs[1].set_ylim([0.81, 1.19])
            axs[1].axhline(y=1., c="black", ls="--", lw=0.7)
            axs[1].axhline(y=1.2, c="black", ls="dotted", lw=0.5)
            axs[1].axhline(y=0.8, c="black", ls="dotted", lw=0.5)
            axs[1].tick_params(axis="both", labelsize=FONTSIZE)

            plt.xlabel(dataset.observables[dim]["tex_label"], fontsize=FONTSIZE)
            plt.savefig(pp, format="pdf", bbox_inches="tight", pad_inches=0.05)
            plt.close()

        # plot correlations
        if isinstance(dataset, Z_2j_dataset):

            correlation_names = ["dimuon_pt", "dimuon_mass", "jet_seperation"]
            correlation_texlabels = [r"$p_{T, \mu \mu }$", r"$m_{\mu \mu }$", r"$\Delta R_{j_1,j_2}$"]
            correlation_functions = [dataset.calculate_dimuon_pt, dataset.calculate_dimuon_mass, dataset.calculate_jet_seperation]
            correlation_units = ["GeV", "GeV", ""]
            correlation_yscales = ["linear", "linear", "linear"]
            correlations_bins = [torch.linspace(0, 200, 50+1), torch.linspace(81, 101, 50+1), torch.linspace(0, 8, 50+1)]

            for i in range(len(correlation_names)):
                bins = correlations_bins[i]
                hist_rec, _ = np.histogram(correlation_functions[i](dataset.rec.cpu()), density=True, bins=bins)
                hist_gen, _ = np.histogram(correlation_functions[i](dataset.gen.cpu()), density=True, bins=bins)

                if len(dataset.unfolded.shape) == 2:
                    hist_unfolded, _ = np.histogram(correlation_functions[i](dataset.unfolded.cpu()), density=True, bins=bins).unsqueeze(0)
                else:
                    n_bayesian_samples = dataset.unfolded.shape[0]
                    hist_unfolded = np.stack([np.histogram(correlation_functions[i](dataset.unfolded[sample, :, :].cpu()), density=True, bins=bins)[0] for sample in range(n_bayesian_samples)])
                if weights is not None:
                    hist_unfolded_reweighted, _ = np.histogram(correlation_functions[i](dataset.unfolded.cpu()), weights=weights, density=True, bins=bins)

                fig1, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.00})
                fig1.tight_layout(pad=0.6, w_pad=0.5, h_pad=0.6, rect=(0.07, 0.06, 0.99, 0.95))

                # histogram
                axs[0].step(bins[1:], hist_rec, label="Rec", linewidth=1.0, where="post", color="blue")
                axs[0].step(bins[1:], hist_gen, label="Gen", linewidth=1.0, where="post", color="black")
                axs[0].step(bins[1:], hist_unfolded[-1], label="Unfolded", linewidth=1.0, where="post", color="red")
                if n_bayesian_samples > 1:
                    axs[0].step(bins[1:], hist_unfolded.mean(axis=0), label="Unfolded Mean", linewidth=1.0, where="post", color="green")

                if n_bayesian_samples > 1:
                    axs[0].fill_between(bins[1:], hist_unfolded.mean(axis=0) - hist_unfolded.std(axis=0), hist_unfolded.mean(axis=0) + hist_unfolded.std(axis=0), alpha=0.2, color="green", step="post")
                    axs[0].fill_between(bins[1:], hist_unfolded[-1] - hist_unfolded.std(axis=0), hist_unfolded[-1] + hist_unfolded.std(axis=0), alpha=0.2, color="red", step="post")

                if weights is not None:
                    axs[0].step(bins[1:], hist_unfolded_reweighted, label="Unfolded (reweighted)", linewidth=1.0, where="post", color="green")

                axs[0].set_yscale(correlation_yscales[i])
                axs[0].legend(frameon=False, fontsize=FONTSIZE)
                axs[0].set_ylabel("Normalized", fontsize=FONTSIZE)
                axs[0].tick_params(axis="both", labelsize=FONTSIZE)

                axs[1].step(bins[1:], hist_unfolded[-1] / hist_gen, color="red", where="post")
                if n_bayesian_samples > 1:
                    axs[1].step(bins[1:], hist_unfolded.mean(axis=0) / hist_gen, color="green", where="post")
                if weights is not None:
                    axs[1].step(bins[1:], hist_unfolded_reweighted / hist_gen, color="green", where="post")


                if n_bayesian_samples > 1:
                    axs[1].fill_between(bins[1:], (hist_unfolded.mean(axis=0)-hist_unfolded.std(axis=0)) / hist_gen, (hist_unfolded.mean(axis=0)+hist_unfolded.std(axis=0)) / hist_gen, alpha=0.2, color="green", step="post")
                    axs[1].fill_between(bins[1:], (hist_unfolded[-1]-hist_unfolded.std(axis=0)) / hist_gen, (hist_unfolded[-1]+hist_unfolded.std(axis=0)) / hist_gen, alpha=0.2, color="red", step="post")

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


def migration_plots(path, dataset):
    dims = dataset.gen.shape[-1]
    with PdfPages(path) as pp:
        for dim in range(dims):
            bins = dataset.observables[dim]["bins"]
            fig, axs = plt.subplots(1,2, figsize=(10,4))
            # True migration
            axs[0].hist2d(dataset.gen[:, dim].cpu().numpy(), dataset.rec[:, dim].cpu().numpy(), density=True, bins=bins, rasterized=True)
            axs[0].set_title(f"True " + dataset.observables[dim]["tex_label"], fontsize=FONTSIZE)
            axs[0].set_xlabel("Rec", fontsize=FONTSIZE)
            axs[0].set_ylabel("Gen", fontsize=FONTSIZE)

            # Model migration
            axs[1].hist2d(dataset.unfolded[-1, :,  dim].cpu().numpy(), dataset.rec[:, dim].cpu().numpy(), density=True, bins=bins, rasterized=True)
            axs[1].set_title(f"Model " + dataset.observables[dim]["tex_label"], fontsize=FONTSIZE)
            axs[1].set_xlabel("Rec", fontsize=FONTSIZE)
            axs[1].set_ylabel("Unfold", fontsize=FONTSIZE)
            plt.savefig(pp, format="pdf", bbox_inches="tight")
            plt.close()


def single_event_plots(path, dataset):
    n_events = dataset.single_event_unfolded.shape[0]
    n_unfoldings = dataset.single_event_unfolded.shape[1]
    dims = dataset.single_event_unfolded.shape[2]

    with PdfPages(path) as pp:
        for dim in range(dims):
            for event in range(n_events):
                event_rec = dataset.rec[event, dim]
                event_gen = dataset.gen[event, dim]
                event_unfoldings = dataset.single_event_unfolded[event, :, dim]

                bins = dataset.observables[dim]["bins"]
                plt.axvline(event_rec.cpu(), label="Event Rec", color="blue")
                plt.axvline(event_gen.cpu(), label="Event Gen", color="orange")
                plt.hist(event_unfoldings.cpu(), density=True, histtype="step", bins=bins, label="Unfoldings", color="green")
                plt.hist(dataset.gen[:, dim].cpu(), density=True, bins=bins, label="Full Gen", alpha=0.2)
                plt.legend()
                plt.title(f"Event {event}", fontsize=FONTSIZE)
                plt.ylabel("Normalized", fontsize=FONTSIZE)
                plt.xlabel(dataset.observables[dim]["tex_label"], fontsize=FONTSIZE)
                plt.savefig(pp, format="pdf", bbox_inches="tight")
                plt.close()


def loss_plot(path, train_loss, val_loss=None):
    n_epochs = len(train_loss)
    epochs = np.arange(n_epochs)
    plt.plot(epochs[2:], train_loss[2:], label="Train loss")
    if val_loss is not None:
        assert len(val_loss) == n_epochs
        plt.plot(epochs[2:], val_loss[2:], label="Val loss")
    plt.legend(fontsize=FONTSIZE)
    plt.xlabel("Epoch", fontsize=FONTSIZE)
    plt.ylabel("Loss", fontsize=FONTSIZE)
    plt.savefig(path, format="pdf", bbox_inches="tight")
    plt.close()


def time_evolution_plots(path, dataset):
    dims = dataset.gen.shape[-1]
    with PdfPages(path) as pp:
        for dim in range(dims):
            for i in range(200):
                if torch.sign(dataset.unfolded[i, 0, dim]) == torch.sign(dataset.unfolded[i, -1, dim]):
                    plt.plot(np.arange(dataset.unfolded.shape[1]), dataset.unfolded[i, :, dim].cpu().numpy(), color='r')
                else:
                    plt.plot(np.arange(dataset.unfolded.shape[1]), dataset.unfolded[i, :, dim].cpu().numpy(), color='k')
            plt.savefig(pp, format="pdf", bbox_inches="tight")
            plt.close()


"""
mask = dataset.rec[:, dim] > 0
hist_rec, _ = np.histogram(dataset.rec[mask, dim].cpu(), density=True, bins=bins)
hist_gen, _ = np.histogram(dataset.gen[mask, dim].cpu(), density=True, bins=bins)
hist_unfolded, _ = np.histogram(dataset.unfolded[mask, -1, dim].cpu(), density=True, bins=bins)

fig1, axs = plt.subplots(1, 1)#, sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.00})
fig1.tight_layout(pad=0.6, w_pad=0.5, h_pad=0.6, rect=(0.07, 0.06, 0.99, 0.95))

# histogram
axs.step(bins[1:], hist_rec, label="Rec", linewidth=1.0, where="post")
axs.step(bins[1:], hist_gen, label="Gen", linewidth=1.0, where="post")
axs.step(bins[1:], hist_unfolded, label="Unfolded", linewidth=1.0, where="post")
axs.set_yscale(dataset.observables[dim]["yscale"])

axs.legend(frameon=False, fontsize=FONTSIZE)
axs.set_ylabel("Normalized", fontsize=FONTSIZE)
axs.tick_params(axis="both", labelsize=FONTSIZE)

# ratio panel
#axs[1].step(bins[1:], hist_unfolded / hist_gen)
#axs[1].set_ylabel(r"$\frac{\mathrm{Model}}{\mathrm{True}}$", fontsize=FONTSIZE)
#axs[1].set_yticks([0.9, 1, 1.1])
#axs[1].set_ylim([0.81, 1.19])
#axs[1].axhline(y=1., c="black", ls="--", lw=0.7)
#axs[1].axhline(y=1.2, c="black", ls="dotted", lw=0.5)
#axs[1].axhline(y=0.8, c="black", ls="dotted", lw=0.5)
#axs[1].tick_params(axis="both", labelsize=FONTSIZE)

plt.xlabel(dataset.observables[dim]["tex_label"], fontsize=FONTSIZE)
plt.savefig(pp, format="pdf", bbox_inches="tight", pad_inches=0.05)
plt.close()
"""
