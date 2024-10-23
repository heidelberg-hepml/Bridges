import torch
import numpy as np
import h5py


class Omnifold:

    def __init__(self, params):
        self.params = params

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.init_dataset(test=False)
        self.apply_preprocessing()
        self.init_observables()

    def init_dataset(self, test=False):
        n_data = self.params["n_train"] if not test else self.params["n_test"]
        path = self.params["path_train"] if not test else self.params["path_test"]

        with h5py.File(path, "r") as f:
            gen = np.array(f["hard"])[:n_data, [0, 2, 5, 1, 3, 4]]
            rec = np.array(f["reco"])[:n_data, [0, 2, 5, 1, 3, 4]]

        self.gen = torch.tensor(gen).float()#.to(self.device)
        self.rec = torch.tensor(rec).float()#.to(self.device)

    def apply_preprocessing(self, reverse=False):

        if not reverse:
            # add noise to the jet multiplicity to smear out the integer structure
            noise = torch.rand_like(self.rec[:, 1]) - 0.5
            self.rec[:, 1] = self.rec[:, 1] + noise
            noise = torch.rand_like(self.gen[:, 1]) - 0.5
            self.gen[:, 1] = self.gen[:, 1] + noise

            # apply special preprocessing to the groomed momentum fraction
            rec5 = self.rec[:, 5]
            noise = torch.rand(size=rec5.shape, device=rec5.device)/1000. * 3 + 0.097
            rec5 = torch.where(rec5 < 0.1, noise, rec5)
            rec5 = rec5.log()
            self.rec_shift = (rec5.max() + rec5.min())/2.
            rec5 = rec5-self.rec_shift
            self.rec_factor = max(rec5.max(), -1 * rec5.min())*1.001
            rec5 = rec5/self.rec_factor
            rec5 = torch.erfinv(rec5)
            self.rec[:, 5] = rec5

            gen5 = self.gen[:, 5]
            noise = torch.rand(size=gen5.shape, device=gen5.device)/1000. * 3 + 0.097
            gen5 = torch.where(gen5 < 0.1, noise, gen5)
            gen5 = gen5.log()
            self.gen_shift = (gen5.max() + gen5.min())/2.
            gen5 = gen5-self.gen_shift
            self.gen_factor = max(gen5.max(), -1 * gen5.min())*1.001
            gen5 = gen5/self.gen_factor
            gen5 = torch.erfinv(gen5)
            self.gen[:, 5] = gen5

            # standardize events
            self.rec_mean = self.rec.mean(0)
            self.rec_std = self.rec.std(0)
            self.gen_mean = self.gen.mean(0)
            self.gen_std = self.gen.std(0)

            self.gen = ((self.gen - self.gen_mean)/self.gen_std).to(self.device)
            self.rec = ((self.rec - self.rec_mean)/self.rec_std).to(self.device)

            if self.params.get("shift_gen", False):
                print("    Shifting gen")
                #self.gen = self.gen[:, [1, 2, 3, 4, 5, 0]]
                self.gen = self.gen[:, [2, 3, 4, 5, 0, 1]]
        else:
            if not hasattr(self, "rec_mean"):
                raise ValueError("Trying to run reverse preprocessing before forward preprocessing")

            if self.params.get("shift_gen", False):
                self.gen = self.gen[:, [4, 5, 0, 1, 2, 3]]

            # undo standardization
            self.gen = self.gen.cpu() * self.gen_std + self.gen_mean
            self.rec = self.rec.cpu() * self.rec_std + self.rec_mean

            # round jet multiplicity back to integers
            self.rec[..., 1] = torch.round(self.rec[..., 1])
            self.gen[..., 1] = torch.round(self.gen[..., 1])

            # undo the special preprocessing to the groomed momentum fraction
            rec5 = self.rec[..., 5]
            rec5 = torch.erf(rec5) * self.rec_factor + self.rec_shift
            rec5 = rec5.exp()
            rec5 = torch.where(rec5 < 0.1, 0, rec5)
            self.rec[..., 5] = rec5

            gen5 = self.gen[..., 5]
            gen5 = torch.erf(gen5) * self.gen_factor + self.gen_shift
            gen5 = gen5.exp()
            gen5 = torch.where(gen5 < 0.1, 0, gen5)
            self.gen[..., 5] = gen5

            if hasattr(self, "unfolded"):
                if self.params.get("shift_gen", False):
                    self.unfolded = self.unfolded[..., [4, 5, 0, 1, 2, 3]]
                self.unfolded = self.unfolded * self.gen_std + self.gen_mean
                self.unfolded[..., 1] = torch.round(self.unfolded[..., 1])

                unfolded5 = self.unfolded[..., 5]
                unfolded5 = torch.erf(unfolded5) * self.gen_factor + self.gen_shift
                unfolded5 = unfolded5.exp()
                unfolded5 = torch.where(unfolded5 < 0.1, 0, unfolded5)
                self.unfolded[..., 5] = unfolded5

            if hasattr(self, "single_event_unfolded"):
                if self.params.get("shift_gen", False):
                    self.single_event_unfolded = self.single_event_unfolded[..., [4, 5, 0, 1, 2, 3]]
                self.single_event_unfolded = self.single_event_unfolded * self.gen_std + self.gen_mean
                self.single_event_unfolded[..., 1] = torch.round(self.single_event_unfolded[..., 1])

                unfolded5 = self.single_event_unfolded[..., 5]
                unfolded5 = torch.erf(unfolded5) * self.gen_factor + self.gen_shift
                unfolded5 = unfolded5.exp()
                unfolded5 = torch.where(unfolded5 < 0.1, 0, unfolded5)
                self.single_event_unfolded[..., 5] = unfolded5

    def init_observables(self):
        self.observables = []

        self.observables.append({
            "tex_label": r"\text{Jet mass } m",
            "bins": torch.linspace(1, 60, 50 + 1),
            "yscale": "linear"
        })
        self.observables.append({
            "tex_label": r"\text{Jet multiplicity } N",
            "bins": torch.arange(3.5, 60.5),
            "yscale": "linear"
        })
        self.observables.append({
            "tex_label": r"$\text{N-subjettiness ratio } \tau_{21}$",
            "bins": torch.linspace(0.1, 1.1, 50 + 1),
            "yscale": "linear"
        })
        self.observables.append({
            "tex_label": r"\text{Jet width } w",
            "bins": torch.linspace(0, 0.6, 50 + 1),
            "yscale": "log"
        })
        self.observables.append({
            "tex_label": r"$\text{Groomed mass }\log \rho$",
            "bins": torch.linspace(-14, -2, 50 + 1),
            "yscale": "linear"
        })
        self.observables.append({
            "tex_label": r"$\text{Groomed momentum fraction }z_g$",
            "bins": torch.linspace(0.05, 0.55, 50 + 1),
            "yscale": "log"
        })








