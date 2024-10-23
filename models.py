import torch
import torch.nn as nn
from torchdiffeq import odeint
import time
from transformer import Transformer
from vblinear import VBLinear

class Model(nn.Module):
    def __init__(self, dims_x, dims_c, params):
        super().__init__()
        self.dims_x = dims_x
        self.dims_c = dims_c
        self.params = params
        self.bayesian = self.params.get("bayesian", False)
        if self.bayesian:
            self.bayesian_factor = self.params.get("bayesian_factor", 1.e-3)
            print(f"Using Bayesian factor {self.bayesian_factor}")

    def init_network(self):

        if self.params.get("network", "MLP") == "MLP":
            if self.bayesian:
                layer_class = VBLinear
            else:
                layer_class = nn.Linear
            layers = []
            layers.append(layer_class(self.dims_in, self.params["internal_size"]))
            layers.append(nn.ReLU())
            for _ in range(self.params["hidden_layers"]):
                layers.append(layer_class(self.params["internal_size"], self.params["internal_size"]))
                layers.append(nn.ReLU())
            layers.append(layer_class(self.params["internal_size"], self.dims_x))
            self.network = nn.Sequential(*layers)
            self.network.bayesian_layers = [layer for layer in layers if isinstance(layer, VBLinear)]
        elif self.params.get("network", "MLP") == "Transformer":
            self.network = Transformer(self.dims_x, self.dims_c, self.params)
            self.network.bayesian_layers = [self.network.final_layer if isinstance(self.network.final_layer, VBLinear) else None]
    # Overwrite in child class
    def sample(self, c):
        pass

    # Overwrite in child class
    def batch_loss(self, x, c, weight):
        pass

    def train(self, data_x, data_c, weights=None):
        if weights is None:
            weights = torch.ones((data_x.shape[0])).to(data_x.dtype).to(data_x.device)

        self.network.train()

        split = int(len(data_x) * 0.8)
        trainset = torch.utils.data.TensorDataset(data_x[:split], data_c[:split], weights[:split])
        valset = torch.utils.data.TensorDataset(data_x[split:], data_c[split:], weights[split:])

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.params["batch_size"],
                                             shuffle=True)
        valloader = torch.utils.data.DataLoader(valset, batch_size=self.params["batch_size"],
                                                  shuffle=True)

        n_epochs = self.params["n_epochs"]
        lr = self.params["lr"]
        optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(trainloader) * n_epochs)
        print(f"Training for {n_epochs} epochs with lr {lr}")
        t0 = time.time()
        self.trainlosses = []
        self.vallosses = []
        if self.bayesian:
            self.kl_losses = []
            self.mse_losses = []
        for epoch in range(n_epochs):
            trainlosses = []
            vallosses = []
            kl_losses = []
            mse_losses = []
            for i, batch in enumerate(trainloader):
                x_hard, x_reco, weight = batch
                optimizer.zero_grad()
                loss = self.batch_loss(x_hard, x_reco, weight)
                if self.bayesian:
                    kl_loss = sum([layer.kl() for layer in self.network.bayesian_layers])
                    kl_loss = kl_loss * self.bayesian_factor * 1./split
                    kl_losses.append(kl_loss.item())
                    mse_losses.append(loss.item())
                    loss = loss + kl_loss
                if loss < 1000:
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    trainlosses.append(loss.item())
                else:
                    print(f"    Skipped update in epoch {epoch}, batch {i}, loss is", loss.item())

            for i, batch in enumerate(valloader):
                with torch.no_grad():
                    x_hard, x_reco, weight = batch
                    loss = self.batch_loss(x_hard, x_reco, weight)
                    if loss < 1000:
                        vallosses.append(loss.item())
                    #else:
                    #    print(f"    Skipped update in epoch {epoch}, batch {i}, loss is", loss.item())

            avg_trainloss = torch.tensor(trainlosses).mean().item()
            avg_valloss = torch.tensor(vallosses).mean().item()
            if self.bayesian:
                avg_kl_loss = torch.tensor(kl_losses).mean().item()
                avg_mse_loss = torch.tensor(mse_losses).mean().item()
            self.trainlosses.append(avg_trainloss)
            self.vallosses.append(avg_valloss)
            if self.bayesian:
                self.kl_losses.append(avg_kl_loss)
                self.mse_losses.append(avg_mse_loss)
            if epoch % int(n_epochs / 5) == 0:
                print(f"    Finished epoch {epoch} with trainloss {avg_trainloss}, valloss {avg_valloss} after time {round(time.time() - t0, 1)}")
                if self.bayesian:
                    print(f"    KL loss: {avg_kl_loss}, MSE loss: {avg_mse_loss}")
        print(f"    Finished final epoch {epoch} with trainloss {avg_trainloss}, valloss {avg_valloss} after time {round(time.time() - t0, 1)}")
        if self.bayesian:
            print(f"    KL loss: {avg_kl_loss}, MSE loss: {avg_mse_loss}")

    def evaluate(self, data_c):
        self.network.eval()
        all_predictions = []
        with torch.no_grad():
            n_bayesian_samples = self.params.get("n_bayesian_samples", 1) if self.bayesian else 1
            for sample in range(n_bayesian_samples):
                if self.bayesian:
                    for layer in self.network.bayesian_layers:
                        layer.map = False
                        layer.reset_random()
                predictions = []
                batches = torch.split(data_c, self.params["batch_size_sample"])
                t0 = time.time()
                for i, batch in enumerate(batches):
                    unfold = self.sample(batch).detach().cpu()
                    predictions.append(unfold)
                    t1 = time.time()
                    if i == 0:
                        print(f"    Total batches: {len(batches)}. First batch took {round(t1-t0, 1)} seconds")
                t2 = time.time()
                print(f"    Finished bayesian sample {sample+1} of {n_bayesian_samples} after {round(t2 - t0, 1)} seconds")
                all_predictions.append(torch.cat(predictions))

            if self.bayesian:
                for layer in self.network.bayesian_layers:
                    layer.map = True
                predictions = []
                batches = torch.split(data_c, self.params["batch_size_sample"])
                t0 = time.time()
                for i, batch in enumerate(batches):
                    unfold = self.sample(batch).detach().cpu()
                    predictions.append(unfold)
                    t1 = time.time()
                    if i == 0:
                        print(f"    Total batches: {len(batches)}. First batch took {round(t1-t0, 1)} seconds")
                t2 = time.time()
                print(f"    Finished MAP sample after {round(t2 - t0, 1)} seconds")
                all_predictions.append(torch.cat(predictions))

        print("All predictions shape:", torch.stack(all_predictions, dim=0).shape)
        return torch.stack(all_predictions, dim=0)

    def single_event_unfolding(self, data_c):
        self.network.eval()
        predictions = []
        with torch.no_grad():
            t0 = time.time()
            for i, event in enumerate(data_c):
                condition = event.repeat(self.params["batch_size_sample"], 1)
                unfold = self.sample(condition).detach().cpu()
                predictions.append(unfold)
                t1 = time.time()
                if i == 0:
                    print(f"    Total events: {len(data_c)}. First event took {round(t1-t0, 1)} seconds")
        predictions = torch.stack(predictions, dim=0)
        return predictions


class CFM(Model):
    def __init__(self, dims_x, dims_c, params):
        super().__init__(dims_x, dims_c, params)
        self.dims_in = self.dims_x + self.dims_c + 1
        self.init_network()

    def sample(self, c):
        batch_size = c.size(0)
        dtype = c.dtype
        device = c.device

        def net_wrapper(t, x_t):
            t = t * torch.ones_like(x_t[:, [0]], dtype=dtype, device=device)
            v = self.network(torch.cat([t, x_t, c], dim=-1))
            return v

        x_0 = torch.randn((batch_size, self.dims_x)).to(device, dtype=dtype)
        #x_t = odeint(func=net_wrapper, y0=x_0, t=torch.linspace(0., 1., 1000).to(device, dtype=dtype))
        #return torch.swapaxes(x_t, 0, 1)
        x_t = odeint(func=net_wrapper,
                     y0=x_0,
                     t=torch.Tensor([0., 1.]).to(device, dtype=dtype),
                     rtol=1e-5,
                     atol=1e-7)
        return x_t[-1]

    def batch_loss(self, x, c, weight):
        x_0 = torch.randn((x.size(0), self.dims_x)).to(x.device)
        t = torch.rand((x.size(0), 1)).to(x.device)
        x_t = (1 - t) * x_0 + t * x
        x_t_dot = x - x_0
        v_pred = self.network(torch.cat([t, x_t, c], dim=-1))
        loss = ((v_pred - x_t_dot) ** 2 * weight.unsqueeze(-1)).mean()
        return loss


class Didi(Model):
    def __init__(self, dims_x, dims_c, params):
        super().__init__(dims_x, dims_c, params)
        self.cond_x1 = self.params.get("cond_x1", False)
        self.noise_scale = self.params.get("noise_scale", 1.e-2)
        self.dims_in = self.dims_x + 1 + int(self.cond_x1) * self.dims_c
        self.init_network()

    def sample(self, x_1):
        dtype = x_1.dtype
        device = x_1.device

        def net_wrapper(t, x_t):
            t = t * torch.ones_like(x_t[:, [0]], dtype=dtype, device=device)
            if self.cond_x1:
                f = self.network(torch.cat([t, x_t, x_1], dim=-1))
            else:
                f = self.network(torch.cat([t, x_t], dim=-1))
            return f

        if self.noise_scale > 0:
            steps = torch.linspace(1, 0, self.params.get("n_steps", 1000))
            pair_steps = zip(steps[1:], steps[:-1])
            pair_steps = pair_steps
            x_t = x_1.detach()
            #x_t_trajectory = [x_t]
            for tprev, t in pair_steps:
                drift = net_wrapper(t, x_t)
                pred_x0 = x_t - t * drift
                x_t = (t - tprev) / t * pred_x0 + tprev / t * x_t
                x_t += (self.noise_scale * tprev * (t - tprev) / t).sqrt() * torch.randn_like(x_t)
                #x_t_trajectory.append(x_t)
            #return torch.stack(x_t_trajectory, dim=1)
            return x_t
        else:
            x_t = odeint(func=net_wrapper,
                         y0=x_1,
                         t=torch.Tensor([1., 0.],).to(device, dtype=dtype),
                         rtol=1e-5,
                         atol=1e-7)
            return x_t[-1]

    def batch_loss(self, x_0, x_1, weight):
        noise = torch.randn_like(x_0)
        t = torch.rand((x_0.size(0), 1)).to(x_0.device)
        if self.noise_scale > 0:
            x_t = (1 - t) * x_0 + t * x_1 + (self.noise_scale*t*(1.-t)).sqrt() * noise
            f = (x_t-x_0)/t
        else:
            x_t = (1 - t) * x_0 + t * x_1
            f = x_1 - x_0
        if self.cond_x1:
            f_pred = self.network(torch.cat([t, x_t, x_1], dim=-1))
        else:
            f_pred = self.network(torch.cat([t, x_t], dim=-1))
        loss = ((f_pred - f) ** 2 * weight.unsqueeze(-1))
        return loss.mean()
