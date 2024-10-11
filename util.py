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
