# some functions were adapted from
# https://github.com/mrdbourke/pytorch-deep-learning/blob/main/going_modular/going_modular/engine.py
import json
import random
from datetime import datetime as dt
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from torchvision import transforms
from tqdm.auto import tqdm

import load_data_functions as ld
import metrics as mt
from pretrained_models.emonet_v1 import AdaptedEmoNet


def train_step(model: nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> tuple[float, float, float]:
    """
    Train the given model for one epoch using the provided data.

    :param model: The neural network model to train.
    :param dataloader: DataLoader providing batches of training data.
    :param loss_fn: The loss function to compute the loss.
    :param optimizer: The optimizer to update the model parameters.
    :param device: The device (CPU or GPU) to perform computations.
    :return: Tuple containing average training loss, CCC (Concordance Correlation Coefficient),
        and PCC (Pearson Correlation Coefficient) over the training dataset.
    """
    # set the model to train mode
    model.train()

    train_loss, train_ccc, train_pcc = 0.0, 0.0, 0.0

    for batch, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device).float(), labels.to(device).float()

        # forward pass
        outputs = model(inputs).flatten()

        # compute loss
        loss = loss_fn(labels, outputs)
        train_loss += loss.item()

        # compute ccc and pcc
        train_ccc += mt.torch_CCC(labels, outputs).item()
        train_pcc += mt.torch_PCC(labels, outputs)

        # backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # compute average metrics
    num_batches = len(dataloader)
    train_loss /= num_batches
    train_ccc /= num_batches
    train_pcc /= num_batches

    return train_loss, train_ccc, train_pcc


def val_step(model: nn.Module,
             dataloader: torch.utils.data.DataLoader,
             loss_fn: nn.Module,
             device: torch.device) -> Tuple[float, float, float]:
    """
    Evaluate the given model on the validation dataset.

    :param model: The neural network model to evaluate.
    :param dataloader: DataLoader providing batches of validation data.
    :param loss_fn: The loss function to compute the validation loss.
    :param device: The device (CPU or GPU) to perform computations.
    :return: Tuple containing average validation loss, CCC (Concordance Correlation Coefficient),
        and PCC (Pearson Correlation Coefficient) over the validation dataset.
    """
    # set the model to evaluation mode
    model.eval()

    val_loss, val_ccc, val_pcc = 0.0, 0.0, 0.0

    with torch.inference_mode():
        # iterate over validation batches
        for batch, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            # forward pass
            outputs = model(inputs).flatten()

            # compute loss
            loss = loss_fn(labels, outputs)
            val_loss += loss.item()

            # compute ccc and pcc
            val_ccc += mt.torch_CCC(labels, outputs).item()
            val_pcc += mt.torch_PCC(labels, outputs)

    # Compute average metrics
    num_batches = len(dataloader)
    val_loss /= num_batches
    val_ccc /= num_batches
    val_pcc /= num_batches

    return val_loss, val_ccc, val_pcc


def plot_loss_curves(results: dict) -> None:
    """
    Plot loss, CCC, and PCC curves over epochs.

    :param results: Dictionary containing training and validation metrics over epochs.
    It should have keys: "train_loss", "val_loss", "train_ccc", "val_ccc", "train_pcc", "val_pcc".
    """
    # extract data from results dictionary
    loss = results["train_loss"]
    val_loss = results["val_loss"]

    ccc = results["train_ccc"]
    val_ccc = results["val_ccc"]

    pcc = results["train_pcc"]
    val_pcc = results["val_pcc"]

    epochs = range(len(results["train_loss"]))

    # plot the figure
    plt.figure(figsize=(15, 4))

    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot CCC
    plt.subplot(1, 3, 2)
    plt.plot(epochs, ccc, label="train_ccc")
    plt.plot(epochs, val_ccc, label="val_ccc")
    plt.title("CCC")
    plt.xlabel("Epochs")
    plt.legend()

    # plot PCC
    plt.subplot(1, 3, 3)
    plt.plot(epochs, pcc, label="train_pcc")
    plt.plot(epochs, val_pcc, label="val_pcc")
    plt.title("PCC")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()


def load_model_for_eval(architecture_path: str = None,
                        state_dict_path: str = None,
                        dropout: float = 0.261807) -> torch.nn.Module:
    """
    Load a model for evaluation, including architecture and state dictionary.

    :param architecture_path: Path to the model architecture file. If None, the path to the pretrained EfficientNet
    B0 model is used.
    :param state_dict_path: Path to the model state dictionary file. If None, the path to the trained EfficientNet
    B0 model is used.
    :param dropout: The dropout probability to be used. Default is 0.28882.
    :return: The loaded model ready for evaluation.
    """
    # set default architecture path if None is provided
    if architecture_path is None:
        architecture_path = "pretrained_models/enet_b0_8_best_vgaf.pt"

    # load the model architecture
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(architecture_path, map_location=torch.device(device))

    # update the classifier layer of the model
    set_seeds()

    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features=1280,
                  out_features=1,
                  bias=True),
        nn.Tanh()
    ).to(device)

    # load the state dictionary into the model
    if state_dict_path is None:
        state_dict_path = "pretrained_models/final_enet.pth"

    state_dict = torch.load(state_dict_path, map_location=torch.device(device))
    model.load_state_dict(state_dict)

    return model


def predict_valence(transformed_image: torch.tensor,
                    model: torch.nn.Module = None) -> float:
    """
    Predict valence for teh given transformed image using the specified model.

    :param transformed_image: The transformed image tensor to be evaluated.
    :param model: The model to use for prediction. If None, a default model is loaded.
    :return: The predicted valence value.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformed_image = transformed_image.to(device)
    if model is None:
        model = load_model_for_eval()

    # set the model to evaluation mode and predict the valence
    model.eval()
    with torch.no_grad():
        valence_pred = model(transformed_image)

    return valence_pred.item()


def pred_and_plot_image(model: torch.nn.Module,
                        image_path: str,
                        image_size: Tuple[int, int] = (224, 224),
                        transform: transforms.Compose = None,
                        device: torch.device = None) -> None:
    """
    Predict the label for an image using the provided model and plot the image with the predicted label.

    :param model: The trained neural network model.
    :param image_path: The path to the image.
    :param image_size: The size to which the image should be resized. Default is (224, 224).
    :param transform: Transformations to be applied to the image.
    If None, default transformations will be applied.
    :param device: The device (CPU or GPU) to perform computations. Default is CPU.
    """
    # open the image
    img = Image.open(image_path)

    # define image transformations
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    # move model to device
    model.to(device)

    # set model to evaluation mode
    model.eval()
    with torch.inference_mode():
        # apply transformations on the image
        transformed_image = transform(img).unsqueeze(dim=0)

        # predict label for the image
        target_image_pred = model(transformed_image.to(device))

    # get actual label for the image
    target_label = ld.get_scaled_label(image_path)

    # plot the image with predicted label
    plt.figure()
    plt.imshow(img)
    plt.title(f"Pred: {target_image_pred.item():.3f} | Actual: {target_label}")
    plt.axis(False)
    plt.show()


def pred_and_plot_n_images(model: torch.nn.Module,
                           val_path: str,
                           num_images_to_plot: int = 3,
                           image_size: Tuple[int, int] = (224, 224),
                           transform: transforms.Compose = None,
                           device: torch.device = None):
    """
    Predict and plot multiple images from a validation dataset using the provided model.

    :param model: The trained neural network model.
    :param val_path: The path to the validation dataset.
    :param num_images_to_plot: The number of images to randomly select and plot. Default is 3.
    :param image_size: The size to which the images should be resized. Default is (224, 224).
    :param transform: Transformations to be applied to the images. If None, default transformations will be applied.
    :param device: The device (CPU or GPU) to perform computations. Default is CPU.
    """
    # get list of all image paths from the validation dataset
    val_path_list = list(Path(val_path).glob("*/*.png"))

    # randomly select "num_images_to_plot" image paths
    test_image_path_sample = random.sample(population=val_path_list, k=num_images_to_plot)

    # predict and plot each selected image
    for image_path in test_image_path_sample:
        pred_and_plot_image(model=model,
                            image_path=image_path,
                            image_size=image_size,
                            transform=transform,
                            device=device)


# Set seeds
def set_seeds(seed: int = 42):
    """
    Set random seeds for torch operations.

    :param seed: Random seed to set. Defaults to 42.
    """
    random.seed(seed)
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)


def create_writer(experiment_phase: str,
                  model_name: str,
                  extra: str = None,
                  project_path: str = None) -> SummaryWriter:
    """
    Create a SummaryWriter for TensorBoard logging.

    :param experiment_phase: Phase of the experiment (e.g., model_selection, tuning...).
    :param model_name: Name of the model.
    :param extra: Additional identifier (default: None).
    :param project_path: The path to the current project / working directory.
    :return: SummaryWriter object for logging
    """
    now = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
    runs_folder = Path("runs") if project_path is None else Path(project_path) / "runs"

    # create log directory
    if extra:
        log_dir = runs_folder / experiment_phase / model_name / extra / now
    else:
        log_dir = runs_folder / experiment_phase / model_name / now

    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=str(log_dir))


def load_enet_baseline(device: torch.device,
                       show_summary_before: bool = False) -> nn.Module:
    """
    Load an EfficientNet-B0 model with a custom classifier head.

    :param device: Device to load the model onto.
    :param show_summary_before: Whether to display model summary before modification. Default is False.
    :return: Loaded EfficientNet-B0 model with custom classifier head.
    """
    model = torchvision.models.efficientnet_b0().to(device)

    # set the seeds
    set_seeds()

    # get a summary of the model if specified
    if show_summary_before:
        summary(model=model,
                input_size=(32, 3, 256, 256),
                col_names=["input_size", "output_size", "num_params", "trainable"],
                col_width=20,
                row_settings=["var_names"]
                )

    # change the classifier head
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),  # to keep the dropout layer that is also there in the original model
        nn.Linear(in_features=1280,
                  out_features=1,
                  bias=True),
        nn.Tanh()  # yields values in the range -1 to 1
    ).to(device)

    # move model to the specified device
    model.to(device)

    print(f"[INFO] Created a new baseline EfficientNet-B0 model.")
    return model


def load_enet_best(device: torch.device, dropout: float = 0.0,
                   project_path: str = None) -> nn.Module:
    """
    Load the pretrained EfficientNet-B0 (Savchenko et al., 2022) and adapt its classifier head
    for regression.
    Reference: https://github.com/av-savchenko/face-emotion-recognition/tree/main/models/affectnet_emotions

    :param device: The device (CPU or GPU) to load the model onto.
    :param dropout: Dropout probability for the classifier head. Default is 0.
    :param project_path: The path to the current project / working directory.
    :return: The loaded and pretrained EfficientNet-B0 model with the adapted classifier head for regression.
    """
    # load a version of the EfficientNet-B0 pretrained by Savchenko et al. (2022)
    model_dir = "pretrained_models/enet_b0_8_best_vgaf.pt"
    model_path = Path(model_dir) if project_path is None else Path(project_path) / model_dir
    model = torch.load(model_path)

    # freeze layers except for the classifier head
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    # set seeds
    set_seeds()

    # change the classifier head for regression
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features=1280,
                  out_features=1,
                  bias=True),
        nn.Tanh()
    ).to(device)

    # move model to the specified device
    model.to(device)

    print(f"[INFO] Created a new pretrained EfficientNet-B0 model.")
    return model


def load_adapted_emonet(device: torch.device, project_path: str = None) -> AdaptedEmoNet:
    """
    Load a modified version of EmoNet with a customized head and freeze specific layers except for valence_fc.
    Reference: https://github.com/face-analysis/emonet/blob/master/emonet/models/emonet.py

    :param device: The device (CPU or GPU) to load the model onto.
    :param project_path: The path to the current project / working directory.
    :return: The loaded adapted EmoNet model.
    """
    # instantiate the adapted EmoNet with a customized heat
    set_seeds()
    model = AdaptedEmoNet(n_expression=5)

    # load pre-trained weights
    weights_dir = "pretrained_models/emonet_5.pth"
    weights_path = Path(weights_dir) if project_path is None else Path(project_path) / weights_dir
    model.load_state_dict(torch.load(weights_path), strict=False)

    # freeze layers except for valence_fc
    for name, param in model.named_parameters():
        if "valence_fc" not in name:
            param.requires_grad = False

    # move model to the specified device
    model.to(device)
    print(f"[INFO] Created a new baseline EmoNet model.")

    return model


def get_loss_optimizer(loss_fn_str: str, optimizer_str: str,
                       model: nn.Module, learning_rate: float) -> Tuple[nn.Module, torch.optim.Optimizer]:
    """
    Get loss function and optimizer based on provided strings.

    :param loss_fn_str: String representing the loss function (e.g., 'MSE' for Mean Squared Error, 'MAE' for Mean
     Absolute Error).
    :param optimizer_str: String representing the optimizer (e.g., 'Adam').
    :param model: The neural network model.
    :param learning_rate: Learning rate for the optimizer.
    :return: Tuple containing the loss function and optimizer.
    """
    # initialize loss function based on loss_fn_str
    if loss_fn_str == "MSE":
        loss_fn = nn.MSELoss()
    elif loss_fn_str == "MAE":
        loss_fn = nn.L1Loss()
    else:
        print("No matching loss function found. Loss_fn_str must be in ['MSE', 'MAE']")
        loss_fn = None

    # initialize optimizer based on optimizer_str
    if optimizer_str == "Adam":
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    else:
        print("No matching optimizer found. Optimizer_str must be in ['Adam']")
        optimizer = None

    return loss_fn, optimizer


def train(num_epochs: int,
          model: nn.Module,
          train_loader: torch.utils.data.DataLoader,
          val_loader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          device: torch.device,
          writer: SummaryWriter = None,
          early_stopping: bool = False,
          patience: int = 5,
          fold: str = None) -> Dict[str, List[float]]:
    """
    Train a model for a specified number of epochs.

    :param num_epochs: Number of training epochs.
    :param model: The neural network model to be trained.
    :param train_loader: DataLoader for the training dataset.
    :param val_loader: DataLoader for the validation dataset.
    :param loss_fn: Loss function for training.
    :param optimizer: Optimizer for training.
    :param device: Device to perform training on (e.g., 'cuda' or 'cpu').
    :param writer: SummaryWriter object for logging to TensorBoard (default: None).
    :param early_stopping: Whether to apply early stopping (default: False).
    :param patience: Number of epochs to wait for improvement before stopping (default: 5).
    :param fold: Fold that is used as validation dataset during the current raining run (default: None).
    :return: Dictionary containing training and validation metrics for each epoch.
    """
    fold_train_cccs = []
    fold_train_pccs = []
    fold_train_loss = []
    fold_val_cccs = []
    fold_val_pccs = []
    fold_val_loss = []

    best_validation_loss = float("inf")
    counter = 0

    for epoch in tqdm(range(num_epochs)):
        train_loss, train_ccc, train_pcc = train_step(model=model,
                                                      dataloader=train_loader,
                                                      loss_fn=loss_fn,
                                                      optimizer=optimizer,
                                                      device=device)

        val_loss, val_ccc, val_pcc = val_step(model=model,
                                              dataloader=val_loader,
                                              loss_fn=loss_fn,
                                              device=device)

        fold_train_loss.append(train_loss)
        fold_train_cccs.append(train_ccc)
        fold_train_pccs.append(train_pcc)
        fold_val_loss.append(val_loss)
        fold_val_cccs.append(val_ccc)
        fold_val_pccs.append(val_pcc)

        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_ccc: {train_ccc:.4f} | "
            f"train_pcc: {train_pcc:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_ccc: {val_ccc:.4f} | "
            f"val_pcc: {val_pcc:.4f}"
        )

        # track training results if a writer was specified
        if writer is not None:
            main_tag_loss = "Loss" if fold is None else f"Loss_{fold}"
            main_tag_ccc = "CCC" if fold is None else f"CCC_{fold}"
            main_tag_pcc = "PCC" if fold is None else f"PCC_{fold}"

            writer.add_scalars(main_tag=main_tag_loss,
                               tag_scalar_dict={"train_loss": train_loss,
                                                "val_loss": val_loss},
                               global_step=epoch)

            # Add accuracy results to SummaryWriter
            writer.add_scalars(main_tag=main_tag_ccc,
                               tag_scalar_dict={"train_ccc": train_ccc,
                                                "val_ccc": val_ccc},
                               global_step=epoch)
            writer.add_scalars(main_tag=main_tag_pcc,
                               tag_scalar_dict={"train_pcc": train_pcc,
                                                "val_pcc": val_pcc},
                               global_step=epoch)

        # implement early stopping if specified
        if early_stopping:
            if val_loss < best_validation_loss:
                best_validation_loss = val_loss
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping: Training stopped at epoch {epoch + 1}")
                    break

    return {
        "train_loss": fold_train_loss,
        "train_ccc": fold_train_cccs,
        "train_pcc": fold_train_pccs,
        "val_loss": fold_val_loss,
        "val_ccc": fold_val_cccs,
        "val_pcc": fold_val_pccs
    }


def calculate_cv_performance(performance_dict: Dict[str, Dict[str, List[float]]],
                             num_epochs: int,
                             writer: SummaryWriter = None,
                             config: Dict[str, any] = None):
    """
    Calculate and log the cross-validation performance metrics over multiple epochs.

    :param performance_dict: A dictionary where each key corresponds to a fold and the value is another dictionary
    containing the lists of validation and training metrics for each epoch.
    :param num_epochs: The number of epochs to iterate through.
    :param writer: A SummaryWriter for logging metrics.
    :param config: A dictionary containing configuration parameters for logging hyperparameters.
    :return: A dictionary containing lists of averaged validation and training metrics for each epoch.
    """
    # initialize lists to store epoch-wise averages
    epoch_avg_ccc = []
    epoch_avg_pcc = []
    epoch_acg_loss = []
    epoch_avg_train_ccc = []
    epoch_avg_train_pcc = []
    epoch_avg_train_loss = []

    # iterate over epochs
    for epoch in range(num_epochs):
        # extract validation and training metrics for all folds
        fold_val_cccs = [perf["val_ccc"][epoch] for perf in performance_dict.values()]
        fold_val_pccs = [perf["val_pcc"][epoch] for perf in performance_dict.values()]
        fold_val_loss = [perf["val_loss"][epoch] for perf in performance_dict.values()]
        fold_train_cccs = [perf["train_ccc"][epoch] for perf in performance_dict.values()]
        fold_train_pccs = [perf["train_pcc"][epoch] for perf in performance_dict.values()]
        fold_train_loss = [perf["train_loss"][epoch] for perf in performance_dict.values()]

        # calculate mean metrics for validation and training
        mean_val_ccc = np.mean(fold_val_cccs)
        mean_val_pcc = np.mean(fold_val_pccs)
        mean_val_loss = np.mean(fold_val_loss)
        mean_train_ccc = np.mean(fold_train_cccs)
        mean_train_pcc = np.mean(fold_train_pccs)
        mean_train_loss = np.mean(fold_train_loss)

        # append to epoch-wise lists
        epoch_avg_ccc.append(mean_val_ccc)
        epoch_avg_pcc.append(mean_val_pcc)
        epoch_acg_loss.append(mean_val_loss)
        epoch_avg_train_ccc.append(mean_train_ccc)
        epoch_avg_train_pcc.append(mean_train_pcc)
        epoch_avg_train_loss.append(mean_train_loss)

        # if writer exists, add scalars for tensorboard visualization
        if writer is not None:
            writer.add_scalars(main_tag="avg_ccc",
                               tag_scalar_dict={"mean_val_ccc": mean_val_ccc,
                                                "mean_train_ccc": mean_train_ccc},
                               global_step=epoch)
            writer.add_scalars(main_tag="avg_pcc",
                               tag_scalar_dict={"mean_val_pcc": mean_val_pcc,
                                                "mean_train_pcc": mean_train_pcc},
                               global_step=epoch)
            writer.add_scalars(main_tag="avg_loss",
                               tag_scalar_dict={"mean_val_loss": mean_val_loss,
                                                "mean_train_loss": mean_train_loss})

            # log hyperparameters and metrics if config is provided
            if config is not None:
                hparam_dict = {f"{key}": value for key, value in config.items()}
                hparam_dict["epoch"] = epoch
                metric_dict = {"hparam/mean_val_ccc": mean_val_ccc, "hparam/mean_val_pcc": mean_val_pcc}
                writer.add_hparams(hparam_dict, metric_dict, global_step=epoch)

    # print relevant performance results
    print({"mean_ccc": epoch_avg_ccc, "mean_pcc": epoch_avg_pcc})

    # close the writer if it exists
    if writer is not None:
        writer.close()

    return {
        "val_ccc": epoch_avg_ccc,
        "val_pcc": epoch_avg_pcc,
        "val_loss": epoch_acg_loss,
        "train_ccc": epoch_avg_train_ccc,
        "train_pcc": epoch_avg_train_pcc,
        "train_loss": epoch_avg_train_loss,
    }


def pretty_json(hp: dict) -> str:
    """
    Return a JSON representation of the dictionary with proper indentation.

    :param hp: The dictionary to convert to JSON.
    :return: The formatted JSON string.
    """
    json_hp = json.dumps(hp, indent=2)
    return "".join("\t" + line for line in json_hp.splitlines(True))
