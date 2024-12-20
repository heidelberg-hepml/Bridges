import argparse
import yaml
import os
import time
import torch
import torch.nn as nn
from datetime import datetime
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import numpy as np
import h5py

from util import calculate_dimuon_pt, calculate_dimuon_mass, calculate_jet_seperation

def main():
    # read in arguments. path to the folder with the SB results and path to classifier params file
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('params_path')
    args = parser.parse_args()

    # read in classifier params
    with open(args.params_path, 'r') as f:
        classifier_params = yaml.safe_load(f)
    
    # check if conditional (meaning we train on gen and rec as opposed to just gen)
    conditional = classifier_params.get("conditional", False)
    
    # create folder for classifier results
    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    classifier_folder = os.path.join(args.model_path, f"classifier_{'Cond' if conditional else 'Uncond'}_{date_time}")
    os.makedirs(classifier_folder)
    # save classifier params
    with open(os.path.join(classifier_folder, "classifier_params.yaml"), 'w') as f:
        yaml.dump(classifier_params, f)

    # create log file
    log_file = os.path.join(classifier_folder, "classifier_log.txt")
    with open(log_file, 'w') as f:
        f.write(f"Starting classifier for run from dir {args.model_path}\n")
    # redirect print to log file
    original_print = print
    def custom_print(*args, **kwargs):
        original_print(*args, **kwargs)
        with open(log_file, 'a') as f:
            original_print(*args, file=f, **kwargs)
    import builtins
    builtins.print = custom_print

    # look for GPUs
    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"
    print(f"Using device {device}")

    # load the samples 
    if "Z2" in args.momodel_path:
        file_gen = "/remote/gpu07/huetsch/Bridges/SB_results_Z2j/Z2Jet_SB_SC_gen.npy"
        file_rec = "/remote/gpu07/huetsch/Bridges/SB_results_Z2j/Z2Jet_SB_SC_reco.npy"

        if args.model_path == "/remote/gpu07/huetsch/Bridges/SB_Z2j_0511/SB_BS":
            file_samples = "/remote/gpu07/huetsch/Bridges/SB_Z2j_0511/SB_BS/Z2Jet_SB_BS_step50_unfolded.npy"
            
        elif args.model_path == "/remote/gpu07/huetsch/Bridges/SB_Z2j_0511/SB_BS_cond":
            file_samples = "/remote/gpu07/huetsch/Bridges/SB_Z2j_0511/SB_BS_cond/Z2Jet_SB_BS_SC_step50_unfolded.npy"

        elif args.model_path == "/remote/gpu07/huetsch/Bridges/SB_Z2j_0511/SB_long":
            file_samples = "/remote/gpu07/huetsch/Bridges/SB_Z2j_0511/SB_long/Z2Jet_SB_Long_step50_unfolded.npy"

        elif args.model_path == "/remote/gpu07/huetsch/Bridges/SB_Z2j_0511/SB_long_cond":
            file_samples = "/remote/gpu07/huetsch/Bridges/SB_Z2j_0511/SB_long_cond/Z2Jet_SB_Long_SC_step50_unfolded.npy"

        else:
            raise ValueError(f"Unknown model path {args.model_path}")
    
        print(f"Loading samples from {file_samples}")
        unfolded = torch.from_numpy(np.load(file_samples, allow_pickle=True)).float().squeeze() 
        gen = torch.from_numpy(np.load(file_gen, allow_pickle=True)).float().squeeze()
        rec = torch.from_numpy(np.load(file_rec, allow_pickle=True)).float().squeeze()

    elif "OF" in args.model_path:
        if args.model_path == "/remote/gpu07/huetsch/Bridges/SB_results_OF/Conditional":
            file_samples = "/remote/gpu07/huetsch/Bridges/SB_results_OF/SBUnfold_OmniFolddata_largeset_SC_unfolded.npy"
        elif args.model_path == "/remote/gpu07/huetsch/Bridges/SB_results_OF/Unconditional":
            "/remote/gpu07/huetsch/Bridges/SB_results_OF/SBUnfold_OmniFolddata_largeset_unfolded.npy"
            
        print(f"Loading samples from {file_samples}")
        unfolded = torch.from_numpy(np.load(file_samples, allow_pickle=True)).float().squeeze() 
        path_testdata = "/remote/gpu07/huetsch/data/omnifold_data/OmniFold_big/OmniFold_test.h5"
        with h5py.File(path_testdata, "r") as f:
            gen = torch.from_numpy(np.array(f["hard"])[:]).float().squeeze()
            rec = torch.from_numpy(np.array(f["reco"])[:]).float().squeeze()    

    dims_gen = gen.shape[-1]
    dims_rec = rec.shape[-1]

    print(f"Dataset shapes: unfolded {unfolded.shape}, gen {gen.shape}, rec {rec.shape}")

    # if the dataset is Z_2j_dataset, include correlations
    if dims_gen == 22:   
        print("Including correlations")
        dimuon_pt_unfolded = calculate_dimuon_pt(unfolded).unsqueeze(1)
        dimuon_pt_rec = calculate_dimuon_pt(rec).unsqueeze(1)
        dimuon_pt_gen = calculate_dimuon_pt(gen).unsqueeze(1)

        dimuon_mass_unfolded = calculate_dimuon_mass(unfolded).unsqueeze(1)
        dimuon_mass_rec = calculate_dimuon_mass(rec).unsqueeze(1)
        dimuon_mass_gen = calculate_dimuon_mass(gen).unsqueeze(1)   

        jet_seperation_unfolded = calculate_jet_seperation(unfolded).unsqueeze(1)
        jet_seperation_rec = calculate_jet_seperation(rec).unsqueeze(1)
        jet_seperation_gen = calculate_jet_seperation(gen).unsqueeze(1)

        unfolded = torch.cat([unfolded, dimuon_pt_unfolded, dimuon_mass_unfolded, jet_seperation_unfolded], dim=1)
        gen = torch.cat([gen, dimuon_pt_gen, dimuon_mass_gen, jet_seperation_gen], dim=1)
        rec = torch.cat([rec, dimuon_pt_rec, dimuon_mass_rec, jet_seperation_rec], dim=1)

        dims_gen += 3
        dims_rec += 3

    # standardize the data
    mean_gen, std_gen = gen.mean(dim=0), gen.std(dim=0)
    mean_rec, std_rec = rec.mean(dim=0), rec.std(dim=0)

    unfolded = (unfolded - mean_gen) / std_gen
    gen = (gen - mean_gen) / std_gen
    rec = (rec - mean_rec) / std_rec


    # run the classifier for the given number of iterations
    iterations = classifier_params.get("iterations", 1)
    print(f"Running {iterations} iterations")

    for iteration in range(iterations):
        print(f"Iteration {iteration}")

        # build classifier with specified network
        print("Building classifier")
        dims_in = dims_gen + dims_rec if conditional else dims_gen
        if classifier_params["network"] == "MLP":
            layers = []
            layers.append(nn.Linear(dims_in, classifier_params["internal_size"]))
            layers.append(nn.LeakyReLU())
            if classifier_params.get("dropout", 0) > 0:
                layers.append(nn.Dropout(classifier_params["dropout"]))
            for _ in range(classifier_params["hidden_layers"]):
                layers.append(nn.Linear(classifier_params["internal_size"], classifier_params["internal_size"]))
                layers.append(nn.LeakyReLU())
                if classifier_params.get("dropout", 0) > 0:
                    layers.append(nn.Dropout(classifier_params["dropout"]))
            layers.append(nn.Linear(classifier_params["internal_size"], 1))
            classifier = nn.Sequential(*layers).to(device)
        elif classifier_params["network"] == "Transformer":
            classifier = Classification_Transformer(dims_in, classifier_params).to(device)

        # define loss function
        loss_fct = nn.BCEWithLogitsLoss()

        # prepare data
        # for each iteration, sample 80% of the data for training and 20% for validation
        print("Preparing data")
        n_samples = len(unfolded)
        permutation = torch.randperm(n_samples)
        n_train = int(n_samples * 0.8)
        train_indices = permutation[:n_train]
        val_indices = permutation[n_train:]

        # if conditional, train on gen and rec, otherwise just on gen
        if conditional:
            train_true = torch.cat([gen[train_indices], rec[train_indices]], dim=1)
            train_false = torch.cat([unfolded[train_indices], rec[train_indices]], dim=1)
        else:
            train_true = gen[train_indices]
            train_false = unfolded[train_indices]

        # concatenate true and false samples and move to device. prepare labels
        train_data = torch.cat([train_true, train_false], dim=0).to(device)
        train_labels = torch.cat([torch.ones(len(train_true)), torch.zeros(len(train_false))], dim=0).to(device).unsqueeze(1)

        # build validation data and labels
        if conditional:
            val_true = torch.cat([gen[val_indices], rec[val_indices]], dim=1)
            val_false = torch.cat([unfolded[val_indices], rec[val_indices]], dim=1)
        else:
            val_true = gen[val_indices]
            val_false = unfolded[val_indices]
        val_data = torch.cat([val_true, val_false], dim=0).to(device)
        val_labels = torch.cat([torch.ones(len(val_true)), torch.zeros(len(val_false))], dim=0).to(device).unsqueeze(1)
        print(f"Train true shape is {train_true.shape}, train false shape is {train_false.shape}, Val true shape is {val_true.shape}, val false shape is {val_false.shape}")

        
        trainset = torch.utils.data.TensorDataset(train_data, train_labels)
        valset = torch.utils.data.TensorDataset(val_data, val_labels)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=classifier_params["batch_size"],
                                                shuffle=True)
        valloader = torch.utils.data.DataLoader(valset, batch_size=classifier_params["batch_size"],
                                                    shuffle=True)

        # define training parameters and build optimizer and scheduler
        n_epochs = classifier_params["n_epochs"]
        lr = classifier_params["lr"]
        optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(trainloader) * n_epochs)

        # train the classifier
        print(f"Training for {n_epochs} epochs with lr {lr}")
        t0 = time.time()
        trainlosses_epoch = []
        vallosses_epoch = []
        best_valloss = float("inf")
        for epoch in range(n_epochs):
            trainlosses = []
            vallosses = []
            for i, batch in enumerate(trainloader):
                x, label = batch
                optimizer.zero_grad()
                prediction = classifier(x)
                loss = loss_fct(prediction, label)
                loss.backward()
                optimizer.step()
                scheduler.step()
                trainlosses.append(loss.item())
                
            for i, batch in enumerate(valloader):
                with torch.no_grad():
                    x, label = batch
                    prediction = classifier(x)
                    loss = loss_fct(prediction, label)
                    vallosses.append(loss.item())

            avg_trainloss = torch.tensor(trainlosses).mean().item()
            avg_valloss = torch.tensor(vallosses).mean().item()
            if avg_valloss < best_valloss:
                best_valloss = avg_valloss
                torch.save(classifier.state_dict(), os.path.join(classifier_folder, f"classifier_{iteration}.pth"))
            trainlosses_epoch.append(avg_trainloss)
            vallosses_epoch.append(avg_valloss)
            if epoch % int(n_epochs / 5) == 0:
                print(f"    Finished epoch {epoch} with trainloss {avg_trainloss}, valloss {avg_valloss} after time {round(time.time() - t0, 1)}")
        print(f"    Finished final epoch {epoch} with trainloss {avg_trainloss}, valloss {avg_valloss} after time {round(time.time() - t0, 1)}")


        # evaluate classifier
        print("Evaluating classifier")
        classifier.load_state_dict(torch.load(os.path.join(classifier_folder, f"classifier_{iteration}.pth"), weights_only=True))
        with torch.no_grad():
            # prepare test data and labels
            if conditional:
                test_true = torch.cat([gen[:n_samples], rec[:n_samples]], dim=1)
                test_false = torch.cat([unfolded[:n_samples], rec[:n_samples]], dim=1)
            else:
                test_true = gen[:n_samples]
                test_false = unfolded[:n_samples]
            test_data = torch.cat([test_true, test_false], dim=0).to(device)
            test_labels = torch.cat([torch.ones(len(test_true)), torch.zeros(len(test_false))], dim=0).to(device).unsqueeze(1)
            testset = torch.utils.data.TensorDataset(test_data, test_labels)
            testloader = torch.utils.data.DataLoader(testset, batch_size=classifier_params["batch_size_sample"], shuffle=False)

            # predict on test data
            predictions = []
            labels = []
            for batch in testloader:
                x, label = batch
                prediction = classifier(x)
                predictions.append(prediction)
                labels.append(label)


        weights = torch.cat(predictions, dim=0).exp().detach().cpu()
        predictions = torch.sigmoid(torch.cat(predictions, dim=0)).detach().cpu()
        labels = torch.cat(labels, dim=0).detach().cpu()

        torch.save(predictions, os.path.join(classifier_folder, f"classifier_predictions_{iteration}.pt"))
        torch.save(labels, os.path.join(classifier_folder, f"classifier_labels_{iteration}.pt"))

        # make plots
        print("Making plots")

        # plot ROC curve
        fpr, tpr, thresholds = roc_curve(labels, predictions)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(classifier_folder, f"classifier_roc_{iteration}.png"))
        plt.close()

        # plot loss
        plt.figure()
        plt.plot(trainlosses_epoch, label='Train loss')
        plt.plot(vallosses_epoch, label='Validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.savefig(os.path.join(classifier_folder, f"classifier_loss_{iteration}.png"))
        plt.close()

        # plot histograms
        plt.figure()
        plt.hist(predictions[labels == 0], bins=50, alpha=0.5, label='False')
        plt.hist(predictions[labels == 1], bins=50, alpha=0.5, label='True')
        plt.xlabel('Predictions')
        plt.ylabel('Frequency')
        plt.title('Histogram of predictions')
        plt.legend()
        plt.yscale('log')
        plt.savefig(os.path.join(classifier_folder, f"classifier_histograms_{iteration}.png"))
        plt.close()

if __name__ == '__main__':
    main()
