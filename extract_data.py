import numpy as np
import uproot


# read in the data
path_delphes = "/Users/huetsch/Desktop/data/nemostuff/new/newrun3/tag_1_delphes_events.root"
file = uproot.open(path_delphes)
delphes_data = file['Delphes;1']
file.close()

"""
build a mask for events with 2 or more gen jets
"""
GenJet_PT = delphes_data['GenJet.PT'].array()
GenJet_nJet = np.array([len(event) for event in GenJet_PT])
GenJet_2JetMask = GenJet_nJet >= 2

"""
build a mask for events with 2 or more sim jets
"""
SimJet_PT = delphes_data['Jet.PT'].array()
SimJet_nJet = np.array([len(event) for event in SimJet_PT])
SimJet_2JetMask = SimJet_nJet >= 2


"""
build the sim muon mask
events are required to have 2 muons with opposite charge, pT > 25 GeV, and m dimuon in (81,101)
"""

# extract muon kinematics
SimMuon_PT = delphes_data['Muon.PT'].array()
SimMuon_eta = delphes_data['Muon.Eta'].array()
SimMuon_phi = delphes_data['Muon.Phi'].array()
SimMuon_charge = delphes_data['Muon.Charge'].array()

SimMuon_Fullmask = np.zeros(len(SimMuon_PT), dtype=bool)
dimuon_pt_sim = []
dimuon_m_sim = []
for i, event in enumerate(SimMuon_PT):
    # check number of detected muons in event
    n_muons_detected = len(event)

    # if less than 2 muons detected, throw event away
    if n_muons_detected < 2:
        SimMuon_Fullmask[i] = False
        continue

    # if 2 or more muons detected check if they meet pT requirements
    else:
        # check if muon pT > 25GeV
        individiual_pt_mask = np.array(event)[:2] > 25
        # if not both muons pT > 25 GeV, throw event away
        if individiual_pt_mask.sum() < 2:
            SimMuon_Fullmask[i] = False
            continue

        # else check dimuon system
        else:
            # extract muon kinematics
            muon1_pt = SimMuon_PT[i][0]
            muon1_eta = SimMuon_eta[i][0]
            muon1_phi = SimMuon_phi[i][0]
            muon2_pt = SimMuon_PT[i][1]
            muon2_eta = SimMuon_eta[i][1]
            muon2_phi = SimMuon_phi[i][1]

            # calculate dimuon pT and mass
            dimuon_pt_2 = muon1_pt ** 2 + muon2_pt ** 2 + 2 * muon1_pt * muon2_pt * np.cos(muon1_phi - muon2_phi)
            dimuon_mass_2 = 2 * muon1_pt * muon2_pt * ((np.cosh(muon1_eta - muon2_eta) - np.cos(muon1_phi - muon2_phi)))

            dimuon_pt = np.sqrt(dimuon_pt_2)
            dimuon_mass = np.sqrt(dimuon_mass_2)

            dimuon_pt_sim.append(dimuon_pt)
            dimuon_m_sim.append(dimuon_mass)

            # check muons have opposite charge
            muon1_charge = SimMuon_charge[i][0]
            muon2_charge = SimMuon_charge[i][1]
            opposite_charge = muon1_charge != muon2_charge

            # cut dimuon pT>200, mass in (81,101), charge opposite
            if dimuon_pt > 0 and dimuon_mass > 81 and dimuon_mass < 101 and opposite_charge:
                SimMuon_Fullmask[i] = True
            else:
                SimMuon_Fullmask[i] = False

print(SimMuon_Fullmask.sum())


"""
build the gen muon mask
events are required to have 2 muons with opposite charge, pT > 25 GeV, and m dimuon in (81,101)
"""

# at gen level we have to first filter out the final state muons
particle_status = file['Delphes;1']["Particle.Status"].array()
particle_ID = file['Delphes;1']["Particle.PID"].array()
particle_pt = file['Delphes;1']["Particle.PT"].array()
particle_eta = file['Delphes;1']["Particle.Eta"].array()
particle_phi = file['Delphes;1']["Particle.Phi"].array()
particle_charge = file['Delphes;1']["Particle.Charge"].array()

GenMuon_Fullmask = np.zeros(len(SimMuon_PT), dtype=bool)

dimuon_pt_gen = []
dimuon_m_gen = []

for i, event in enumerate(particle_pt):
    status = np.array(particle_status[i])
    ID = np.array(particle_ID[i])

    # extract final state particles
    status_filter = status == 1
    # extract muons and antimuons
    ID_filter = abs(ID) == 13
    # combine filters to get final state muons
    status_and_ID_filter = status_filter * ID_filter

    # if less than 2 muons, throw event away
    if status_and_ID_filter.sum() < 2:
        GenMuon_Fullmask[i] = False
        continue

    else:
        # extract muon pt
        muon1_pt = particle_pt[i][status_and_ID_filter][0]
        muon2_pt = particle_pt[i][status_and_ID_filter][1]

        # check for 2 muons with pT > 25GeV
        if muon1_pt < 25 or muon2_pt < 25:
            GenMuon_Fullmask[i] = False
            continue
        else:
            muon1_eta = particle_eta[i][status_and_ID_filter][0]
            muon2_eta = particle_eta[i][status_and_ID_filter][1]
            muon1_phi = particle_phi[i][status_and_ID_filter][0]
            muon2_phi = particle_phi[i][status_and_ID_filter][1]

            # calculate dimuon pT and mass
            dimuon_pt_2 = muon1_pt ** 2 + muon2_pt ** 2 + 2 * muon1_pt * muon2_pt * np.cos(muon1_phi - muon2_phi)
            dimuon_mass_2 = 2 * muon1_pt * muon2_pt * ((np.cosh(muon1_eta - muon2_eta) - np.cos(muon1_phi - muon2_phi)))

            dimuon_pt = np.sqrt(dimuon_pt_2)
            dimuon_mass = np.sqrt(dimuon_mass_2)

            dimuon_pt_gen.append(dimuon_pt)
            dimuon_m_gen.append(dimuon_mass)

            # check muons have opposite charge
            muon1_charge = particle_charge[i][status_and_ID_filter][0]
            muon2_charge = particle_charge[i][status_and_ID_filter][1]
            opposite_charge = muon1_charge != muon2_charge

            # cut dimuon pT>200, mass in (81,101), charge opposite
            if dimuon_pt > 0 and dimuon_mass > 81 and dimuon_mass < 101 and opposite_charge:
                GenMuon_Fullmask[i] = True
            else:
                GenMuon_Fullmask[i] = False

"""
Combine muon and jet masks from sim and gen
"""
Fullmask = GenJet_2JetMask * SimJet_2JetMask * SimMuon_Fullmask * GenMuon_Fullmask
print(Fullmask.sum())


"""
Extract the sim level kinematics for the accepted events
"""
SimJet_PT = delphes_data['Jet.PT'].array()[Fullmask]
SimJet_eta = delphes_data['Jet.Eta'].array()[Fullmask]
SimJet_phi = delphes_data['Jet.Phi'].array()[Fullmask]
SimJet_m = delphes_data['Jet.Mass'].array()[Fullmask]
SimJet_tau = delphes_data['Jet.Tau[5]'].array()[Fullmask]
SimJet_Const = delphes_data['Jet.Constituents'].array()[Fullmask]

SimMuon_PT = delphes_data['Muon.PT'].array()[Fullmask]
SimMuon_eta = delphes_data['Muon.Eta'].array()[Fullmask]
SimMuon_phi = delphes_data['Muon.Phi'].array()[Fullmask]

Sim_events = np.zeros((Fullmask.sum(), 22))

for i in range(Fullmask.sum()):

    # sort muons by pT
    if SimMuon_PT[i][0] > SimMuon_PT[i][1]:
        SimMuon1_pt = SimMuon_PT[i][0]
        SimMuon1_eta = SimMuon_eta[i][0]
        SimMuon1_phi = SimMuon_phi[i][0]

        SimMuon2_pt = SimMuon_PT[i][1]
        SimMuon2_eta = SimMuon_eta[i][1]
        SimMuon2_phi = SimMuon_phi[i][1]

    else:
        SimMuon1_pt = SimMuon_PT[i][1]
        SimMuon1_eta = SimMuon_eta[i][1]
        SimMuon1_phi = SimMuon_phi[i][1]

        SimMuon2_pt = SimMuon_PT[i][0]
        SimMuon2_eta = SimMuon_eta[i][0]
        SimMuon2_phi = SimMuon_phi[i][0]

    # extract jet kinematics
    SimJet1_pt = SimJet_PT[i][0]
    SimJet1_eta = SimJet_eta[i][0]
    SimJet1_phi = SimJet_phi[i][0]
    SimJet1_m = SimJet_m[i][0]
    SimJet1_N = SimJet_Const[i][0]["fSize"]
    SimJet1_tau1 = SimJet_tau[i][0][0]
    SimJet1_tau2 = SimJet_tau[i][0][1]
    SimJet1_tau3 = SimJet_tau[i][0][2]

    SimJet2_pt = SimJet_PT[i][1]
    SimJet2_eta = SimJet_eta[i][1]
    SimJet2_phi = SimJet_phi[i][1]
    SimJet2_m = SimJet_m[i][1]
    SimJet2_N = SimJet_Const[i][1]["fSize"]
    SimJet2_tau1 = SimJet_tau[i][1][0]
    SimJet2_tau2 = SimJet_tau[i][1][1]
    SimJet2_tau3 = SimJet_tau[i][1][2]

    Sim_events[i] = np.array([
        SimMuon1_pt,
        SimMuon1_eta,
        SimMuon1_phi,
        SimMuon2_pt,
        SimMuon2_eta,
        SimMuon2_phi,
        SimJet1_pt,
        SimJet1_eta,
        SimJet1_phi,
        SimJet1_m,
        SimJet1_N,
        SimJet1_tau1,
        SimJet1_tau2,
        SimJet1_tau3,
        SimJet2_pt,
        SimJet2_eta,
        SimJet2_phi,
        SimJet2_m,
        SimJet2_N,
        SimJet2_tau1,
        SimJet2_tau2,
        SimJet2_tau3,
    ])



"""
Extract the gen level kinematics for the accepted events

here we have to make one additional cuts. Sometimes fastjet clusters the muons into the jets
we only accept events that have 2 muons meeting the requirements and 2 jets meeting the requirements that are dR seperated from the muons
"""

GenJet_PT = delphes_data['GenJet.PT'].array()[Fullmask]
GenJet_eta = delphes_data['GenJet.Eta'].array()[Fullmask]
GenJet_phi = delphes_data['GenJet.Phi'].array()[Fullmask]
GenJet_m = delphes_data['GenJet.Mass'].array()[Fullmask]
GenJet_tau = delphes_data['GenJet.Tau[5]'].array()[Fullmask]
GenJet_Const = delphes_data['GenJet.Particles'].array()[Fullmask]

particle_status = file['Delphes;1']["Particle.Status"].array()[Fullmask]
particle_ID = file['Delphes;1']["Particle.PID"].array()[Fullmask]
particle_pt = file['Delphes;1']["Particle.PT"].array()[Fullmask]
particle_eta = file['Delphes;1']["Particle.Eta"].array()[Fullmask]
particle_phi = file['Delphes;1']["Particle.Phi"].array()[Fullmask]

Gen_events = np.zeros((Fullmask.sum(), 22))

for i in range(Fullmask.sum()):
    status = np.array(particle_status[i])
    ID = np.array(particle_ID[i])
    # extract final state particles
    status_filter = status == 1
    # extract muons and antimuons
    ID_filter = abs(ID) == 13
    # combine filters to get final state muons
    status_and_ID_filter = status_filter * ID_filter

    # extract muon kinematics, sort by pT
    if particle_pt[i][status_and_ID_filter][0] > particle_pt[i][status_and_ID_filter][1]:
        GenMuon1_pt = particle_pt[i][status_and_ID_filter][0]
        GenMuon1_eta = particle_eta[i][status_and_ID_filter][0]
        GenMuon1_phi = particle_phi[i][status_and_ID_filter][0]

        GenMuon2_pt = particle_pt[i][status_and_ID_filter][1]
        GenMuon2_eta = particle_eta[i][status_and_ID_filter][1]
        GenMuon2_phi = particle_phi[i][status_and_ID_filter][1]

    else:
        GenMuon1_pt = particle_pt[i][status_and_ID_filter][1]
        GenMuon1_eta = particle_eta[i][status_and_ID_filter][1]
        GenMuon1_phi = particle_phi[i][status_and_ID_filter][1]

        GenMuon2_pt = particle_pt[i][status_and_ID_filter][0]
        GenMuon2_eta = particle_eta[i][status_and_ID_filter][0]
        GenMuon2_phi = particle_phi[i][status_and_ID_filter][0]

    n_jets = len(GenJet_PT[i])
    jets_accepted = []

    # check all jets, look for 2 jets that are dR seperated from the muons
    for jet in range(n_jets):

        # get jet kinematics
        candidateJet_eta = GenJet_eta[i][jet]
        candidateJet_phi = GenJet_phi[i][jet]

        # calculate dR to the 2 muons to make sure they are not in the jet
        dR_jmu1 = np.sqrt((GenMuon1_eta - candidateJet_eta) ** 2 + (GenMuon1_phi - candidateJet_phi) ** 2)
        dR_jmu2 = np.sqrt((GenMuon2_eta - candidateJet_eta) ** 2 + (GenMuon2_phi - candidateJet_phi) ** 2)

        if dR_jmu1 < 0.4 or dR_jmu2 < 0.4:
            continue
        else:
            jets_accepted.append(jet)

    if len(jets_accepted) >= 2:

        GenJet1_pt = GenJet_PT[i][jets_accepted[0]]
        GenJet1_eta = GenJet_eta[i][jets_accepted[0]]
        GenJet1_phi = GenJet_phi[i][jets_accepted[0]]
        GenJet1_m = GenJet_m[i][jets_accepted[0]]
        GenJet1_N = GenJet_Const[i][jets_accepted[0]]["fSize"]
        GenJet1_tau1 = GenJet_tau[i][jets_accepted[0]][0]
        GenJet1_tau2 = GenJet_tau[i][jets_accepted[0]][1]
        GenJet1_tau3 = GenJet_tau[i][jets_accepted[0]][2]

        GenJet2_pt = GenJet_PT[i][jets_accepted[1]]
        GenJet2_eta = GenJet_eta[i][jets_accepted[1]]
        GenJet2_phi = GenJet_phi[i][jets_accepted[1]]
        GenJet2_m = GenJet_m[i][jets_accepted[1]]
        GenJet2_N = GenJet_Const[i][jets_accepted[1]]["fSize"]
        GenJet2_tau1 = GenJet_tau[i][jets_accepted[1]][0]
        GenJet2_tau2 = GenJet_tau[i][jets_accepted[1]][1]
        GenJet2_tau3 = GenJet_tau[i][jets_accepted[1]][2]

    else:

        GenJet1_pt = np.nan
        GenJet1_eta = np.nan
        GenJet1_phi = np.nan
        GenJet1_m = np.nan
        GenJet1_N = np.nan
        GenJet1_tau1 = np.nan
        GenJet1_tau2 = np.nan
        GenJet1_tau3 = np.nan

        GenJet2_pt = np.nan
        GenJet2_eta = np.nan
        GenJet2_phi = np.nan
        GenJet2_m = np.nan
        GenJet2_N = np.nan
        GenJet2_tau1 = np.nan
        GenJet2_tau2 = np.nan
        GenJet2_tau3 = np.nan

    Gen_events[i] = np.array([
        GenMuon1_pt,
        GenMuon1_eta,
        GenMuon1_phi,
        GenMuon2_pt,
        GenMuon2_eta,
        GenMuon2_phi,
        GenJet1_pt,
        GenJet1_eta,
        GenJet1_phi,
        GenJet1_m,
        GenJet1_N,
        GenJet1_tau1,
        GenJet1_tau2,
        GenJet1_tau3,
        GenJet2_pt,
        GenJet2_eta,
        GenJet2_phi,
        GenJet2_m,
        GenJet2_N,
        GenJet2_tau1,
        GenJet2_tau2,
        GenJet2_tau3,
    ])

# mask out events that didnt meet gen jet dR requirements
deltaR_mask = ~np.isnan(Gen_events.sum(1))
Gen_events_dRfiltered = Gen_events[deltaR_mask]
Sim_events_dRfiltered = Sim_events[deltaR_mask]
print(Gen_events_dRfiltered.shape)

# save final events
np.save("Z_2j_Sim.npy", Sim_events_dRfiltered)
np.save("Z_2j_Gen.npy", Gen_events_dRfiltered)