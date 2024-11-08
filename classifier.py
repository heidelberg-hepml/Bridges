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

from omnifold_dataset import Omnifold
from two_gaussians_dataset import TwoGaussians
from Z_2j_dataset import Z_2j_dataset
from models import CFM, Didi
from plots import marginal_plots, migration_plots, single_event_plots, loss_plot, time_evolution_plots
from transformer import Classification_Transformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('params_path')
    args = parser.parse_args()

    with open(args.params_path, 'r') as f:
        classifier_params = yaml.safe_load(f)
    
    conditional = classifier_params.get("conditional", False)
    
    with open(os.path.join(args.model_path, "params.yaml"), 'r') as f:
        model_params = yaml.safe_load(f)
    
    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    classifier_folder = os.path.join(args.model_path, f"classifier_{'Cond' if conditional else 'Uncond'}_{date_time}")
    os.makedirs(classifier_folder)
    with open(os.path.join(classifier_folder, "classifier_params.yaml"), 'w') as f:
        yaml.dump(classifier_params, f)

    log_file = os.path.join(classifier_folder, "classifier_log.txt")
    with open(log_file, 'w') as f:
        f.write(f"Starting classifier for run from dir {args.model_path}\n")
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

    # get the dataset
    print("Building dataset", model_params["dataset_params"]["type"])
    dataset = eval(model_params["dataset_params"]["type"])(model_params["dataset_params"])
    dataset.init_dataset(test=True)
    # get gen and rec dimension from the dataset
    dims_gen = dataset.gen.shape[-1]
    dims_rec = dataset.rec.shape[-1]

    # build the model
    print("Building model", model_params["model_params"]["type"])
    model = eval(model_params["model_params"]["type"])(dims_gen, dims_rec, model_params["model_params"]).to(device)
    print("Loading model")
    model.load_state_dict(torch.load(os.path.join(args.model_path, "model", f"model.pth"),  map_location=device, weights_only=True))

    # try to load samples, if not possible sample model
    try:
        print("Loading samples")
        file_samples = os.path.join(args.model_path, f"samples.pt")
        dataset.unfolded = torch.load(file_samples, map_location="cpu", weights_only=True)[-1]
    except:
        print("Loading samples failed, sampling model")
        dataset.apply_preprocessing()
        dataset.unfolded = model.evaluate(dataset.rec)[-1]
        dataset.apply_preprocessing(reverse=True)
        torch.save(dataset.unfolded, os.path.join(args.model_path, f"samples.pt"))


    if isinstance(dataset, Z_2j_dataset):   
        
        dimuon_pt_unfolded = dataset.calculate_dimuon_pt(dataset.unfolded).unsqueeze(1)
        dimuon_pt_rec = dataset.calculate_dimuon_pt(dataset.rec).unsqueeze(1)
        dimuon_pt_gen = dataset.calculate_dimuon_pt(dataset.gen).unsqueeze(1)

        dimuon_mass_unfolded = dataset.calculate_dimuon_mass(dataset.unfolded).unsqueeze(1)
        dimuon_mass_rec = dataset.calculate_dimuon_mass(dataset.rec).unsqueeze(1)
        dimuon_mass_gen = dataset.calculate_dimuon_mass(dataset.gen).unsqueeze(1)   

        jet_seperation_unfolded = dataset.calculate_jet_seperation(dataset.unfolded).unsqueeze(1)
        jet_seperation_rec = dataset.calculate_jet_seperation(dataset.rec).unsqueeze(1)
        jet_seperation_gen = dataset.calculate_jet_seperation(dataset.gen).unsqueeze(1)

        dataset.unfolded = torch.cat([dataset.unfolded, dimuon_pt_unfolded, dimuon_mass_unfolded, jet_seperation_unfolded], dim=1)
        dataset.gen = torch.cat([dataset.gen, dimuon_pt_gen, dimuon_mass_gen, jet_seperation_gen], dim=1)
        dataset.rec = torch.cat([dataset.rec, dimuon_pt_rec, dimuon_mass_rec, jet_seperation_rec], dim=1)

        dims_gen += 3
        dims_rec += 3

    #mean_unfolded, std_unfolded = dataset.unfolded.mean(dim=0), dataset.unfolded.std(dim=0)
    mean_gen, std_gen = dataset.gen.mean(dim=0), dataset.gen.std(dim=0)
    mean_rec, std_rec = dataset.rec.mean(dim=0), dataset.rec.std(dim=0)

    dataset.unfolded = (dataset.unfolded - mean_gen) / std_gen
    dataset.gen = (dataset.gen - mean_gen) / std_gen
    dataset.rec = (dataset.rec - mean_rec) / std_rec


    iterations = classifier_params.get("iterations", 1)
    print(f"Running {iterations} iterations")

    for iteration in range(iterations):
        print(f"Iteration {iteration}")
        # build classifier
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

        loss_fct = nn.BCEWithLogitsLoss()

        # prepare data
        print("Preparing data")
        n_samples = len(dataset.unfolded)
        permutation = torch.randperm(n_samples)
        n_train = int(n_samples * 0.8)
        train_indices = permutation[:n_train]
        val_indices = permutation[n_train:]

    
        if conditional:
            train_true = torch.cat([dataset.gen[train_indices], dataset.rec[train_indices]], dim=1)
            train_false = torch.cat([dataset.unfolded[train_indices], dataset.rec[train_indices]], dim=1)
        else:
            train_true = dataset.gen[train_indices]
            train_false = dataset.unfolded[train_indices]
        train_data = torch.cat([train_true, train_false], dim=0).to(device)
        train_labels = torch.cat([torch.ones(len(train_true)), torch.zeros(len(train_false))], dim=0).to(device).unsqueeze(1)

        if conditional:
            val_true = torch.cat([dataset.gen[val_indices], dataset.rec[val_indices]], dim=1)
            val_false = torch.cat([dataset.unfolded[val_indices], dataset.rec[val_indices]], dim=1)
        else:
            val_true = dataset.gen[val_indices]
            val_false = dataset.unfolded[val_indices]
        val_data = torch.cat([val_true, val_false], dim=0).to(device)
        val_labels = torch.cat([torch.ones(len(val_true)), torch.zeros(len(val_false))], dim=0).to(device).unsqueeze(1)

        trainset = torch.utils.data.TensorDataset(train_data, train_labels)
        valset = torch.utils.data.TensorDataset(val_data, val_labels)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=classifier_params["batch_size"],
                                                shuffle=True)
        valloader = torch.utils.data.DataLoader(valset, batch_size=classifier_params["batch_size"],
                                                    shuffle=True)

        n_epochs = classifier_params["n_epochs"]
        lr = classifier_params["lr"]
        optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(trainloader) * n_epochs)
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
            if conditional:
                test_true = torch.cat([dataset.gen, dataset.rec], dim=1)
                test_false = torch.cat([dataset.unfolded, dataset.rec], dim=1)
            else:
                test_true = dataset.gen
                test_false = dataset.unfolded
            test_data = torch.cat([test_true, test_false], dim=0).to(device)
            test_labels = torch.cat([torch.ones(len(test_true)), torch.zeros(len(test_false))], dim=0).to(device).unsqueeze(1)
            testset = torch.utils.data.TensorDataset(test_data, test_labels)
            testloader = torch.utils.data.DataLoader(testset, batch_size=classifier_params["batch_size_sample"], shuffle=False)

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
