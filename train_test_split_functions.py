import random
import shutil
from pathlib import Path
from typing import Tuple, List

import load_data_functions as ld


def copy_folders(source: Path, dest: Path, folders: List[Path]):
    """
    Copy selected folders from the source directory to the destination directory.

    :param source: Path to the source directory.
    :param dest: Path to the destination directory.
    :param folders: List of folder paths to be copied.
    :return: Number of folders successfully copied.
    """
    folder_names = [path.name for path in folders]

    # check if the destination exists, else create it
    if not dest.exists():
        dest.mkdir(parents=True, exist_ok=True)

    copied_folders = 0

    # iterate over the selected folders
    for folder in folder_names:
        source_path = source / folder
        dest_path = dest / folder

        try:
            # copy the folder with its files
            shutil.copytree(source_path, dest_path)
            copied_folders += 1
            if copied_folders == 1:
                print("Copying folders...")
        except PermissionError as e:
            print(f"Error when copying {folder}: {e}")
        except FileExistsError as e:
            print(f"Error when copying {folder}: {e}")

    print(f"{copied_folders} folders were successfully copied to {dest}.")


def find_all_ims(folder_list: List[Path]) -> List[Path]:
    """
    Find all image files within the specified list of folders.

    :param folder_list: List of Path objects representing folders.
    :return: List of Path objects representing image files.
    """
    all_ims = [list(folder.glob("*.png")) for folder in folder_list]
    flat = [item for sublist in all_ims for item in sublist]
    return flat


def find_correct_percentage(desired_percentage: float,
                            folder_paths: List[Path],
                            start: int = 0,
                            validation_fold: bool = True,
                            train_path: Path = Path("data/train_test_split/train")) -> Tuple[List[Path], List[Path]]:
    """
    Find a selection of folder paths that approximately matches the desired percentage of total images.

    :param desired_percentage: The desired percentage of total images to be selected.
    :param folder_paths: List of folder paths to select from.
    :param start: The starting seed value for random selection (default: 0).
    :param validation_fold: Flag indicating whether to use validation fold for image count (default: True).
    :param train_path: Path to the training data folder (default: "data/train_test_split/train").
    :return: A tuple containing selected folder paths and not selected folder paths.
    """
    if validation_fold:
        folder_paths_train = ld.get_folder_paths(train_path)
        image_count = len(find_all_ims(folder_paths_train))
        to_select = round(desired_percentage * len(folder_paths_train))
    else:
        image_count = len(find_all_ims(folder_paths))
        to_select = round(desired_percentage * len(folder_paths))

    def test_various_seeds(start: int, to_select=to_select) -> List[Path]:
        """
        Test various seeds to find a selection of folders that matches the desired percentage.

        :param start: The starting seed value for random selection.
        :param to_select: The number of folders to select.
        :return: A list of selected folder paths if found, otherwise None.
        """
        for seed in range(start, start + 100):
            random.seed(seed)
            selected_dirs = random.sample(folder_paths, to_select)
            selected_images = find_all_ims(selected_dirs)
            actual_percentage = len(selected_images) / image_count
            if round(actual_percentage, 2) == desired_percentage:
                print(f"For the seed = {seed}, the percentage of selected images is approximately {desired_percentage}")
                return selected_dirs
        if start == 10000:
            print(
                f"It was not possible to find a split where the percentage of selected images is approximately {desired_percentage}")
            print(f"The current fold now contains {actual_percentage} of all folders.")
            return selected_dirs

    selected_folders = test_various_seeds(start)
    while selected_folders is None:
        start += 100
        selected_folders = test_various_seeds(start)

    not_selected = list(set(folder_paths) - set(selected_folders))
    return selected_folders, not_selected
