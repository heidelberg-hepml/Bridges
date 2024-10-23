import torch
import numpy as np


class TwoGaussians:

    def __init__(self, params):
        self.params = params

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.init_dataset(test=False)
        self.apply_preprocessing()
        self.init_observables()

    def init_dataset(self, test=False):
        n_data = self.params["n_train"] if not test else self.params["n_test"]
        mu = self.params["mu"]
        sig = self.params["sigma"]

        #gen = torch.cat([torch.randn((n_data, 1))*sig+mu, torch.randn((n_data, 1))*sig-mu])
        #gen = gen[torch.randperm(2*n_data)]
        #rec = -gen

        x_data = np.concatenate([np.random.normal(mu, sig, (n_data, 1)), np.random.normal(-mu, sig, (n_data, 1))])
        np.random.shuffle(x_data)
        target = -x_data

        self.gen = torch.tensor(target).float().to(self.device)
        self.rec = torch.tensor(x_data).float().to(self.device)

    def apply_preprocessing(self, reverse=False):
        pass

    def init_observables(self):
        self.observables = []

        boundary = self.params["mu"] + 5 * self.params["sigma"]

        self.observables.append({
            "tex_label": r"x_1",
            "bins": torch.linspace(-boundary, boundary, 50),
            "yscale": "linear"
        })
