import torch
import numpy as np
from util import *



class Z_2j_dataset:

    def __init__(self, params):
        self.params = params

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.init_dataset(test=False)
        self.apply_preprocessing()
        self.init_observables()

    def init_dataset(self, test=False):
        n_data = self.params["n_train"]
        path_gen = self.params["path_gen"]
        path_sim = self.params["path_sim"]

        if test:
            gen = np.load(path_gen)[n_data:]
            rec = np.load(path_sim)[n_data:]
        else:
            gen = np.load(path_gen)[:n_data]
            rec = np.load(path_sim)[:n_data]

        self.gen = torch.tensor(gen).float()
        self.rec = torch.tensor(rec).float()

    def apply_preprocessing(self, reverse=False):

        if not reverse:
            """
            # add noise to the jet multiplicity to smear out the integer structure
            noise = torch.rand_like(self.rec[:, 1]) - 0.5
            self.rec[:, 1] = self.rec[:, 1] + noise
            noise = torch.rand_like(self.gen[:, 1]) - 0.5
            self.gen[:, 1] = self.gen[:, 1] + noise
            """
            #set muon masses to zero
            muon_mass = torch.full((len(self.gen), ), 0.)#.to(self.device)

            muon1_gen_pT_eta_phi_m = torch.stack([self.gen[:, 0],
                                                  self.gen[:, 1],
                                                  self.gen[:, 2],
                                                  muon_mass], dim=-1)
            muon2_gen_pT_eta_phi_m = torch.stack([self.gen[:, 3],
                                                  self.gen[:, 4],
                                                  self.gen[:, 5],
                                                  muon_mass], dim=-1)
            dimuon_mass_gen = invariant_mass(muon1_gen_pT_eta_phi_m, muon2_gen_pT_eta_phi_m)
            dimuon_mass_gen = breit_wigner_forward(dimuon_mass_gen, peak_position=91, width=1)
            self.gen[:, 0] = dimuon_mass_gen

            muon1_rec_pT_eta_phi_m = torch.stack([self.rec[:, 0],
                                                  self.rec[:, 1],
                                                  self.rec[:, 2],
                                                  muon_mass], dim=-1)
            muon2_rec_pT_eta_phi_m = torch.stack([self.rec[:, 3],
                                                  self.rec[:, 4],
                                                  self.rec[:, 5],
                                                  muon_mass], dim=-1)
            dimuon_mass_rec = invariant_mass(muon1_rec_pT_eta_phi_m, muon2_rec_pT_eta_phi_m)
            dimuon_mass_rec = breit_wigner_forward(dimuon_mass_rec, peak_position=91, width=1)
            self.rec[:, 0] = dimuon_mass_rec

            # standardize events
            self.rec_mean = self.rec.mean(0)
            self.rec_std = self.rec.std(0)
            self.gen_mean = self.gen.mean(0)
            self.gen_std = self.gen.std(0)

            self.gen = ((self.gen - self.gen_mean)/self.gen_std).to(self.device)
            self.rec = ((self.rec - self.rec_mean)/self.rec_std).to(self.device)

        else:
            if not hasattr(self, "rec_mean"):
                raise ValueError("Trying to run reverse preprocessing before forward preprocessing")

            # undo standardization
            self.gen = self.gen.cpu() * self.gen_std + self.gen_mean
            self.rec = self.rec.cpu() * self.rec_std + self.rec_mean
            """
            # round jet multiplicity back to integers
            self.rec[..., 1] = torch.round(self.rec[..., 1])
            self.gen[..., 1] = torch.round(self.gen[..., 1])
            """
            dimuon_mass_gen = breit_wigner_reverse(self.gen[..., 0], peak_position=91, width=1)
            muon1_eta_gen = self.gen[..., 1]
            muon1_phi_gen = self.gen[..., 2]
            muon2_pt_gen = self.gen[..., 3]
            muon2_eta_gen = self.gen[..., 4]
            muon2_phi_gen = self.gen[..., 5]
            muon1_pt_gen = dimuon_mass_gen**2 / (2*muon2_pt_gen*(torch.cosh(muon1_eta_gen-muon2_eta_gen)-torch.cos(muon1_phi_gen-muon2_phi_gen)))
            self.gen[..., 0] = muon1_pt_gen

            dimuon_mass_rec = breit_wigner_reverse(self.rec[..., 0], peak_position=91, width=1)
            muon1_eta_rec = self.rec[..., 1]
            muon1_phi_rec = self.rec[..., 2]
            muon2_pt_rec = self.rec[..., 3]
            muon2_eta_rec = self.rec[..., 4]
            muon2_phi_rec = self.rec[..., 5]
            muon1_pt_rec = dimuon_mass_rec ** 2 / (
                        2 * muon2_pt_rec * (torch.cosh(muon1_eta_rec - muon2_eta_rec) - torch.cos(
                    muon1_phi_rec - muon2_phi_rec)))
            self.rec[..., 0] = muon1_pt_rec

            if hasattr(self, "unfolded"):
                self.unfolded = self.unfolded * self.gen_std.cpu() + self.gen_mean.cpu()
                dimuon_mass_unfolded = breit_wigner_reverse(self.unfolded[..., 0], peak_position=91, width=1)
                muon1_eta_unfolded = self.unfolded[..., 1]
                muon1_phi_unfolded = self.unfolded[..., 2]
                muon2_pt_unfolded = self.unfolded[..., 3]
                muon2_eta_unfolded = self.unfolded[..., 4]
                muon2_phi_unfolded = self.unfolded[..., 5]
                muon1_pt_unfolded = dimuon_mass_unfolded ** 2 / (
                        2 * muon2_pt_unfolded * (torch.cosh(muon1_eta_unfolded - muon2_eta_unfolded) - torch.cos(
                    muon1_phi_unfolded - muon2_phi_unfolded)))
                self.unfolded[..., 0] = muon1_pt_unfolded

            if hasattr(self, "single_event_unfolded"):
                self.single_event_unfolded = self.single_event_unfolded * self.gen_std.cpu() + self.gen_mean.cpu()
                dimuon_mass_unfolded = breit_wigner_reverse(self.single_event_unfolded[..., 0], peak_position=91, width=1)
                muon1_eta_unfolded = self.single_event_unfolded[..., 1]
                muon1_phi_unfolded = self.single_event_unfolded[..., 2]
                muon2_pt_unfolded = self.single_event_unfolded[..., 3]
                muon2_eta_unfolded = self.single_event_unfolded[..., 4]
                muon2_phi_unfolded = self.single_event_unfolded[..., 5]
                muon1_pt_unfolded = dimuon_mass_unfolded ** 2 / (
                        2 * muon2_pt_unfolded * (torch.cosh(muon1_eta_unfolded - muon2_eta_unfolded) - torch.cos(
                    muon1_phi_unfolded - muon2_phi_unfolded)))
                self.single_event_unfolded[..., 0] = muon1_pt_unfolded

    def init_observables(self):
        self.observables = []

        self.observables.append({
            "tex_label": r"$p_{T, \mu_1}$",
            "bins": torch.linspace(20, 150, 50 + 1),
            "yscale": "linear",
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
            "bins": torch.linspace(10, 150, 50 + 1),
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


    def calculate_dimuon_pt(self, dataset):

        muon1_pt = dataset[:, 0]
        muon1_eta = dataset[:, 1]
        muon1_phi = dataset[:, 2]
        muon2_pt = dataset[:, 3]
        muon2_eta = dataset[:, 4]
        muon2_phi = dataset[:, 5]

        dimuon_pt_2 = muon1_pt ** 2 + muon2_pt ** 2 + 2 * muon1_pt * muon2_pt * torch.cos(muon1_phi - muon2_phi)
        return dimuon_pt_2.sqrt()

    def calculate_dimuon_mass(self, dataset):

        muon1_pt = dataset[:, 0]
        muon1_eta = dataset[:, 1]
        muon1_phi = dataset[:, 2]
        muon2_pt = dataset[:, 3]
        muon2_eta = dataset[:, 4]
        muon2_phi = dataset[:, 5]

        dimuon_mass_2 = 2 * muon1_pt * muon2_pt * ((np.cosh(muon1_eta - muon2_eta) - np.cos(muon1_phi - muon2_phi)))
        return dimuon_mass_2.sqrt()

    def calculate_jet_seperation(self, dataset):

        jet1_eta = dataset[:, 7]
        jet1_phi = dataset[:, 8]
        jet2_eta = dataset[:, 15]
        jet2_phi = dataset[:, 16]

        dR_2 = (jet1_eta - jet2_eta)**2 + (jet1_phi - jet2_phi)**2
        return dR_2.sqrt()
