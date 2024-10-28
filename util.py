import torch
import numpy as np
from torch.special import erf, erfinv


def pT_eta_phi_m_2_E_px_py_pz(pT_eta_phi_m):
    pt = pT_eta_phi_m[..., 0]
    eta = pT_eta_phi_m[..., 1]
    phi = pT_eta_phi_m[..., 2]
    m = pT_eta_phi_m[..., 3]

    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)
    E = torch.sqrt(px**2 + py**2 + pz**2 + m**2)
    return torch.stack([E, px, py, pz], dim=-1)


def pT_eta_phi_m_2_E_px_py_pz_single(pt, m, eta, phi):
    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)
    E = torch.sqrt(px**2 + py**2 + pz**2 + m**2)
    return torch.stack([E, px, py, pz], dim=-1)


def E_px_py_pz_2_pT_eta_phi_m(E_px_py_pz):
    E = E_px_py_pz[..., 0]
    px = E_px_py_pz[..., 1]
    py = E_px_py_pz[..., 2]
    pz = E_px_py_pz[..., 3]

    pt = torch.sqrt(px**2 + py**2)
    eta = torch.arctanh(pz / np.sqrt(px ** 2 + py ** 2 + pz ** 2))
    phi = torch.arctan2(py, px)
    m = (E ** 2 - px**2 - py**2 - pz**2).sqrt()
    return torch.stack([pt, eta, phi, m], dim=-1)


def E_px_py_pz_2_pT_eta_phi_m_single(E, px, py, pz):
    pt = torch.sqrt(px ** 2 + py ** 2)
    eta = torch.arctanh(pz / np.sqrt(px ** 2 + py ** 2 + pz ** 2))
    phi = torch.arctan2(py, px)
    m = (E ** 2 - px ** 2 - py ** 2 - pz ** 2).sqrt()
    return torch.stack([pt, eta, phi, m], dim=-1)


def add_pT_eta_phi_m(*particles):
    particles_Eppp = [pT_eta_phi_m_2_E_px_py_pz(p) for p in particles]
    particles_sum = sum(particles_Eppp)
    return E_px_py_pz_2_pT_eta_phi_m(particles_sum)


def invariant_mass(*particles):
    particles_Eppp = [pT_eta_phi_m_2_E_px_py_pz(p) for p in particles]
    particles_sum = sum(particles_Eppp)
    return E_px_py_pz_2_pT_eta_phi_m(particles_sum)[..., -1]


def breit_wigner_forward(events, peak_position, width):
    z1 = 1 / np.pi * np.arctan((events - peak_position) / width) + 0.5
    return np.sqrt(2) * erfinv(2 * z1 - 1)


def breit_wigner_reverse(events, peak_position, width):
    a = events/np.sqrt(2)
    a = erf(a)
    a = 0.5*(a+1)
    a = a -0.5
    a = a*np.pi
    a = np.tan(a)
    a = a*width + peak_position
    return a


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