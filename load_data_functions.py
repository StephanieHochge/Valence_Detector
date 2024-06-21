import json
import os
import zipfile as zf
from pathlib import Path
from typing import Tuple, List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from torchvision.transforms import transforms


def extract_zip_files(dat_path: Path, im_path: Path) -> str:
    """
    If the image path already exists, do nothing.
    If not, create the directory and extract zip files into it.

    :param dat_path: Path to the directory containing the zip files.
    :param im_path: Path to the directory where files will be extracted.
    :return: Action taken message.
    """
    # Check if the image path exists
    if im_path.is_dir():
        action = f"{im_path} directory exists. Zip files were not extracted."
    else:
        # Get all filenames to unzip the files
        filenames = [file for file in dat_path.iterdir() if file.is_file() and file.suffix == '.zip']
        # Extract the zip files
        for file in filenames:
            print(f"Currently extracting file {file}")
            with zf.ZipFile(file, "r") as zip_ref:
                zip_ref.extractall(im_path)
        action = f"Created {im_path} directory and unzipped files."
    return action


def get_folder_paths(root_dir: Path) -> List[Path]:
    """
    Get a list of paths to all directories within the specified root directory.

    :param root_dir: Path to the root directory.
    :return: List of Path objects representing directories.
    """
    return [Path(f.path) for f in os.scandir(root_dir) if f.is_dir()]


def get_image_label(im_path: Path) -> float:
    """
    Retrieve the image label from the corresponding JSON file.

    :param im_path: Path to the image file.
    :return: Image label.
    """
    json_file = im_path.parent / (im_path.parent.stem + ".json")
    frame = im_path.stem
    with open(json_file, "r") as f:
        data = json.load(f)
        im_label = data["frames"][frame]["valence"]
    return im_label


def get_scaled_label(im_path):
    """
    Scale the image label. All labels are in the range of -10 to +10. This function turns them into the range of 0 to
    +1.
    :param im_path:
    :return:
    """
    label = get_image_label(im_path)
    return label / 10


def get_all_image_labels(path_list: List[Path], subset: bool = False) -> List[float]:
    """
    Retrieve labels for all image paths in the specified list.

    :param path_list: List containing image paths.
    :param subset: Whether to limit the number of images (default: False).
    :return: List of image labels.
    """
    labels = []
    n = 0
    for path in path_list:
        n += 1
        label = get_image_label(path)
        labels.append(label)
        if subset:
            if n == 100:
                break
    return labels


def compare_distributions(labels_1, labels_2, labels_1_col: str = None, labels_2_col: str = None,
                          save_results: bool = False, save_path: str = None, show: bool = False):
    """
    Compare the distributions of two sets of labels and visualize them using histograms.

    :param labels_1: List or array-like object containing labels for the first distribution.
    :param labels_2: List or array-like object containing labels for the second distribution.
    :param labels_1_col: Name of the column representing labels for the first distribution (default: "labels_1").
    :param labels_2_col: Name of the column representing labels for the second distribution (default: "labels_2").
    :param save_results: Whether to save the comparison results (default: False).
    :param save_path: Path to save the results (default: None).
    :param show: Whether to display the comparison results (default: False).
    :return: DataFrame containing descriptive statistics of the label distributions.
    """
    # assign default column names if not provided
    if labels_1_col is None:
        labels_1_col = "labels_1"
    if labels_2_col is None:
        labels_2_col = "labels_2"

    # create DataFrames for label distributions
    df_1 = pd.DataFrame({labels_1_col: labels_1})
    df_2 = pd.DataFrame({labels_2_col: labels_2})

    # calculate descriptive statistics
    desc_1 = df_1.describe()
    desc_2 = df_2.describe()

    # concatenate descriptive statistics of both distributions
    desc = pd.concat([desc_1, desc_2], axis=1)

    # plot histograms
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    bins = np.linspace(min(min(labels_1), min(labels_2)), max(max(labels_1), max(labels_2)), 20)
    axes[0].hist(labels_1, bins=bins, color='blue', alpha=0.7)
    axes[0].set_xlabel(labels_1_col)
    axes[0].set_ylabel("count")
    axes[0].set_title(labels_1_col)

    axes[1].hist(labels_2, bins=bins, color='green', alpha=0.7)
    axes[1].set_xlabel(labels_2_col)
    axes[1].set_ylabel("count")
    axes[1].set_title(labels_2_col)

    # save comparison results if specified
    if save_results:
        if save_path is None:
            save_path = f"data/splits_metadata/{labels_1_col}_{labels_2_col}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        desc.to_csv(f"{save_path}/desc.csv", sep=";")
        plt.savefig(f"{save_path}/hist.png")

    # display comparison results if specified
    if show:
        print(desc)
        plt.show()
    return desc


class ImageFolderCustom(Dataset):
    """
    Custom dataset class for loading image data from a directory.
    """

    def __init__(self, targ_dir: Path, transform: transforms.Compose = None) -> None:
        """
        Initialize the dataset by loading image paths and setting up transforms.

        :param targ_dir: Path to the directory containing image files.
        :param transform: Optional transform to be applied on the images. Defaults to None.
        """
        self.targ_dir = targ_dir
        self.paths = list(Path(targ_dir).glob("*/*.png"))
        self.transform = transform

    def load_image(self, index: int) -> Image.Image:
        """
        Load an image from a given index.

        :param index: Index of the image to load.
        :return: Loaded image.
        """
        im_path = self.paths[index]
        return Image.open(im_path)

    def __len__(self) -> int:
        """
        Return the number of images in the dataset.

        :return: Number of images in the dataset.
        """
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, float]:
        """
        Retrieve an image and its corresponding label from the dataset.

        :param index: Index of the image to retrieve.
        :return: Tuple containing the image and its label.
        """
        img = self.load_image(index)
        label = get_scaled_label(self.paths[index])

        # Transform if necessary
        if self.transform:
            return self.transform(img), label
        else:
            return img, label


def select_subset(train_data: torch.utils.data.Dataset, percentage: float):
    """
    Select a subset of the training dataset based on the specified percentage.

    :param train_data: The training dataset.
    :param percentage: The percentage of the training data to select.
    :return: Subset of the training dataset.
    """
    train_indices = list(range(len(train_data)))
    train_subset_indices = train_indices[:int(percentage * len(train_indices))]
    train_subset = Subset(train_data, train_subset_indices)

    return train_subset


def create_dataloaders(train_data: torch.utils.data.Dataset,
                       test_data: torch.utils.data.Dataset,
                       device: torch.device,
                       batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoader objects for training and testing datasets.

    :param train_data: Training dataset.
    :param test_data: Testing or validation dataset.
    :param device: Device to load data onto.
    :param batch_size: Batch size for DataLoader.
    :return: DataLoader objects for training and testing datasets.
    """
    kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {}

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader


def create_cv_datasets(fold_list: List[Path],
                       train_transform: transforms.Compose,
                       val_transform: transforms.Compose = None,
                       train_percentage: float = 1.0,
                       select_one: bool = False,
                       augmentation: bool = False,
                       augment_transform: transforms.Compose = None) -> Dict[str, Dict[str, torch.utils.data.Dataset]]:
    """
    Create training and validation datasets for cross-validation.

    :param fold_list: List of fold paths.
    :param train_transform: Transformation pipeline for the training data.
    :param val_transform: Transformation pipeline for the validation data.
    :param train_percentage: Percentage of training data to use. Defaults to 1.0.
    :param select_one: Whether to select only one fold for validation. Defaults to False.
    :param augmentation: Whether to apply data augmentation to the training set. Defaults to False.
    :param augment_transform: Transformation pipeline for the training data to augment. Default is None.
    :return: Dictionary containing training and validation datasets.
    """
    # use train_transform for validation data if val_transform is not provided
    if val_transform is None:
        val_transform = train_transform

    # initialize dictionary to store datasets
    datasets = {}

    # iterate through each fold
    for validation_fold in fold_list:
        # create validation dataset
        validation_data = ImageFolderCustom(targ_dir=validation_fold, transform=val_transform)

        # create training dataset
        train_folds = fold_list.copy()
        train_folds.remove(validation_fold)
        train_data_list = [ImageFolderCustom(targ_dir=fold, transform=train_transform) for fold in train_folds]
        train_data = ConcatDataset(train_data_list)

        if augmentation:
            augmented_list = [ImageFolderCustom(targ_dir=fold, transform=augment_transform) for fold in train_folds]
            augmented = ConcatDataset(augmented_list)
            train_data = ConcatDataset([train_data, augmented])

        # select a subset of training data if specified
        if train_percentage < 1.0:
            train_data = select_subset(train_data, train_percentage)
            print(f"Selected {train_percentage} of train data.")

        # store datasets in dictionary
        datasets[validation_fold.name] = {"train": train_data, f"validation": validation_data}

        # return if only one fold needs to be selected
        if select_one:
            return datasets

    return datasets
