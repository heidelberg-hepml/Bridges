import torch
import numpy as np
import h5py


class Z_2j_dataset:

    def __init__(self, params):
        self.params = params

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.init_dataset(test=False)
        self.apply_preprocessing()
        self.init_observables()

    def init_dataset(self, test=False):
        n_data = self.params["n_train"] if not test else self.params["n_test"]
        path_gen = self.params["path_gen"]
        path_sim = self.params["path_sim"]

        gen = np.load(path_gen)[:n_data]
        rec = np.load(path_sim)[:n_data]

        self.gen = torch.tensor(gen).float().to(self.device)
        self.rec = torch.tensor(rec).float().to(self.device)

    def apply_preprocessing(self, reverse=False):

        if not reverse:
            """
            # add noise to the jet multiplicity to smear out the integer structure
            noise = torch.rand_like(self.rec[:, 1]) - 0.5
            self.rec[:, 1] = self.rec[:, 1] + noise
            noise = torch.rand_like(self.gen[:, 1]) - 0.5
            self.gen[:, 1] = self.gen[:, 1] + noise
            """
            # standardize events
            self.rec_mean = self.rec.mean(0)
            self.rec_std = self.rec.std(0)
            self.gen_mean = self.gen.mean(0)
            self.gen_std = self.gen.std(0)

            self.gen = (self.gen - self.gen_mean)/self.gen_std
            self.rec = (self.rec - self.rec_mean)/self.rec_std

        else:
            if not hasattr(self, "rec_mean"):
                raise ValueError("Trying to run reverse preprocessing before forward preprocessing")

            # undo standardization
            self.gen = self.gen * self.gen_std + self.gen_mean
            self.rec = self.rec * self.rec_std + self.rec_mean
            """
            # round jet multiplicity back to integers
            self.rec[..., 1] = torch.round(self.rec[..., 1])
            self.gen[..., 1] = torch.round(self.gen[..., 1])
            """
            if hasattr(self, "unfolded"):
                self.unfolded = self.unfolded * self.gen_std.cpu() + self.gen_mean.cpu()
                #self.unfolded[..., 1] = torch.round(self.unfolded[..., 1])

            if hasattr(self, "single_event_unfolded"):
                self.single_event_unfolded = self.single_event_unfolded * self.gen_std.cpu() + self.gen_mean.cpu()
                #self.single_event_unfolded[..., 1] = torch.round(self.single_event_unfolded[..., 1])

    def init_observables(self):
        self.observables = []

        self.observables.append({
            "tex_label": r"$p_{T, \mu_1}$",
            "bins": torch.linspace(20, 150, 50 + 1),
            "yscale": "linear"
        })
        self.observables.append({
            "tex_label": r"$\eta_{\mu_1}$",
            "bins": torch.linspace(-2.5, 2.5, 50 + 1),
            "yscale": "linear"
        })
        self.observables.append({
            "tex_label": r"$\phi_{\mu_1}$",
            "bins": torch.linspace(-3.5, 3.5, 50 + 1),
            "yscale": "linear"
        })
        self.observables.append({
            "tex_label": r"$p_{T, \mu_2}$",
            "bins": torch.linspace(20, 100, 50 + 1),
            "yscale": "linear"
        })
        self.observables.append({
            "tex_label": r"$\eta_{\mu_2}$",
            "bins": torch.linspace(-2.5, 2.5, 50 + 1),
            "yscale": "linear"
        })
        self.observables.append({
            "tex_label": r"$\phi_{\mu_2}$",
            "bins": torch.linspace(-3.5, 3.5, 50 + 1),
            "yscale": "linear"
        })
        self.observables.append({
            "tex_label": r"$p_{T, j_1}$",
            "bins": torch.linspace(20, 200, 50 + 1),
            "yscale": "linear"
        })
        self.observables.append({
            "tex_label": r"$\eta_{j_1}$",
            "bins": torch.linspace(-2.5, 2.5, 50 + 1),
            "yscale": "linear"
        })
        self.observables.append({
            "tex_label": r"$\phi_{j_1}$",
            "bins": torch.linspace(-3.5, 3.5, 50 + 1),
            "yscale": "linear"
        })
        self.observables.append({
            "tex_label": r"$m_{j_1}$",
            "bins": torch.linspace(0, 30, 50 + 1),
            "yscale": "linear"
        })
        self.observables.append({
            "tex_label": r"$N_{j_1}$",
            "bins": torch.arange(0.5, 50.5),
            "yscale": "linear"
        })
        self.observables.append({
            "tex_label": r"$\tau_{1, j_1}$",
            "bins": torch.linspace(-0.1, 0.8, 50 + 1),
            "yscale": "linear"
        })
        self.observables.append({
            "tex_label": r"$\tau_{2, j_1}$",
            "bins": torch.linspace(-0.1, 0.5, 50 + 1),
            "yscale": "linear"
        })
        self.observables.append({
            "tex_label": r"$\tau_{3, j_1}$",
            "bins": torch.linspace(-0.1, 0.4, 50 + 1),
            "yscale": "linear"
        })
        self.observables.append({
            "tex_label": r"$p_{T, j_2}$",
            "bins": torch.linspace(18, 150, 50 + 1),
            "yscale": "linear"
        })
        self.observables.append({
            "tex_label": r"$\eta_{j_2}$",
            "bins": torch.linspace(-2.5, 2.5, 50 + 1),
            "yscale": "linear"
        })
        self.observables.append({
            "tex_label": r"$\phi_{j_2}$",
            "bins": torch.linspace(-3.5, 3.5, 50 + 1),
            "yscale": "linear"
        })
        self.observables.append({
            "tex_label": r"$m_{j_2}$",
            "bins": torch.linspace(0, 30, 50 + 1),
            "yscale": "linear"
        })
        self.observables.append({
            "tex_label": r"$N_{j_2}$",
            "bins": torch.arange(0.5, 40.5),
            "yscale": "linear"
        })
        self.observables.append({
            "tex_label": r"$\tau_{1, j_2}$",
            "bins": torch.linspace(-0.1, 0.8, 50 + 1),
            "yscale": "linear"
        })
        self.observables.append({
            "tex_label": r"$\tau_{2, j_2}$",
            "bins": torch.linspace(-0.1, 0.5, 50 + 1),
            "yscale": "linear"
        })
        self.observables.append({
            "tex_label": r"$\tau_{3, j_2}$",
            "bins": torch.linspace(-0.1, 0.4, 50 + 1),
            "yscale": "linear"
        })







