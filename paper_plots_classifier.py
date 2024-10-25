import math
import sys
import torch
import os
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import time
import matplotlib
import h5py
from matplotlib.colors import LogNorm
from sklearn.metrics import roc_curve, auc

# plotting parameters
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Charter"
plt.rcParams["text.usetex"] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage[bitstream-charter]{mathdesign} \usepackage{amsmath}'
FONTSIZE=16


# helper function to make model container
def make_container(path, label, color):
    container = {
    "label": label,
    "color": color,
    "predictions": torch.load(os.path.join(path, "classifier_predictions.pt"), map_location=torch.device('cpu'), weights_only=True).numpy(),
    "labels": torch.load(os.path.join(path, "classifier_labels.pt"), map_location=torch.device('cpu'), weights_only=True).numpy(),
    }
    return container

# build model containers
print("Building model containers")
path_cfm = "/remote/gpu07/huetsch/Bridges/results/20241022_180947_Z2j_CFM_Bayesian_Transformer_500e/classifier_Uncond_20241024_120257"
path_didi = "/remote/gpu07/huetsch/Bridges/results/20241022_180647_Z2j_Didi_0_Bayesian_Transformer_500e/classifier_Uncond_20241024_120257"
path_didiCond = "/remote/gpu07/huetsch/Bridges/results/20241022_184746_Z2j_DidiCond_1e1_Bayesian_Transformer_500e/classifier_Uncond_20241024_120138"

path_cfm = "/remote/gpu07/huetsch/Bridges/results/20241022_180947_Z2j_CFM_Bayesian_Transformer_500e/classifier_Cond_20241024_131134"
path_didi = "/remote/gpu07/huetsch/Bridges/results/20241022_180647_Z2j_Didi_0_Bayesian_Transformer_500e/classifier_Cond_20241024_131134"
path_didiCond = "/remote/gpu07/huetsch/Bridges/results/20241022_184746_Z2j_DidiCond_1e1_Bayesian_Transformer_500e/classifier_Cond_20241024_133603"


container_cfm = make_container(path_cfm, "CFM", "#A52A2A")
container_didi = make_container(path_didi, "Didi", "#008000")
container_didiCond = make_container(path_didiCond, "Cond. Didi", "#FFA500")

fpr_cfm, tpr_cfm, thresholds_cfm = roc_curve(container_cfm["labels"], container_cfm["predictions"])
roc_auc_cfm = auc(fpr_cfm, tpr_cfm)
fpr_didi, tpr_didi, thresholds_didi = roc_curve(container_didi["labels"], container_didi["predictions"])
roc_auc_didi = auc(fpr_didi, tpr_didi)
fpr_didiCond, tpr_didiCond, thresholds_didiCond = roc_curve(container_didiCond["labels"], container_didiCond["predictions"])
roc_auc_didiCond = auc(fpr_didiCond, tpr_didiCond)

plt.figure()
plt.plot(fpr_cfm, tpr_cfm, label='CFM (area = %0.2f)' % roc_auc_cfm, color=container_cfm["color"])
plt.plot(fpr_didi, tpr_didi, label='Didi (area = %0.2f)' % roc_auc_didi, color=container_didi["color"])
plt.plot(fpr_didiCond, tpr_didiCond, label='Cond. Didi (area = %0.2f)' % roc_auc_didiCond, color=container_didiCond["color"])
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Joint Classifier ROC')
plt.legend(loc="lower right")
plt.savefig("paperplots/classifier_Z2j_Cond_roc.png")
plt.close()

bins = np.linspace(-0.03, 1.03, 50)
cfm_predictions_true = container_cfm["predictions"][container_cfm["labels"] == 1]
cfm_predictions_false = container_cfm["predictions"][container_cfm["labels"] == 0]
didi_predictions_true = container_didi["predictions"][container_didi["labels"] == 1]
didi_predictions_false = container_didi["predictions"][container_didi["labels"] == 0]
didiCond_predictions_true = container_didiCond["predictions"][container_didiCond["labels"] == 1]
didiCond_predictions_false = container_didiCond["predictions"][container_didiCond["labels"] == 0]

hist_cfm_true, _ = np.histogram(cfm_predictions_true, bins=bins, density=True)
hist_cfm_false, _ = np.histogram(cfm_predictions_false, bins=bins, density=True)
hist_didi_true, _ = np.histogram(didi_predictions_true, bins=bins, density=True)
hist_didi_false, _ = np.histogram(didi_predictions_false, bins=bins, density=True)
hist_didiCond_true, _ = np.histogram(didiCond_predictions_true, bins=bins, density=True)
hist_didiCond_false, _ = np.histogram(didiCond_predictions_false, bins=bins, density=True)

plt.figure()
plt.step(bins[:-1], hist_cfm_true, where='post', label='CFM True', color=container_cfm["color"], alpha=1)
plt.step(bins[:-1], hist_cfm_false, where='post', label='CFM False', color=container_cfm["color"], alpha=1, linestyle="--")
plt.step(bins[:-1], hist_didi_true, where='post', label='Didi True', color=container_didi["color"], alpha=1)
plt.step(bins[:-1], hist_didi_false, where='post', label='Didi False', color=container_didi["color"], alpha=1, linestyle="--")
plt.step(bins[:-1], hist_didiCond_true, where='post', label='Cond. Didi True', color=container_didiCond["color"], alpha=1)
plt.step(bins[:-1], hist_didiCond_false, where='post', label='Cond. Didi False', color=container_didiCond["color"], alpha=1, linestyle="--")
plt.xlabel('Predictions')
plt.ylabel('Normalized')
plt.title('Joint Classifier distribution')
plt.legend()
plt.yscale('log')
plt.savefig("paperplots/classifier_Z2j_Cond_histograms.png")
plt.close()
