import glob
import json
import os.path
from collections import defaultdict

import click
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from data import process_data
from deepgaze_pytorch import DeepGazeIII
from utils import predict, AlternationAlgorithm
from evaluation import EvaluationFunctions


@click.group()
def cli():
    pass


@cli.command()
@click.option("--image_folders", "-i", type=click.Path(exists=True))
@click.option("--csv_folders", "-c", type=click.Path(exists=True))
@click.option("--device", "-d", default="cuda", help="Device to use for training")
@click.option(
    "--number_of_fixations",
    "-n",
    default=10,
    help="Number of fixations to generate, "
    "if zero it will be same as "
    "the number of fixations in GT",
)
@click.option("--width", "-w", type=int, help="Width of the image")
@click.option("--height", "-h", type=int, help="Height of the image")
@click.option(
    "--radius",
    "-r",
    type=float,
    help="Radius of the fixation it should be a float between 0 and 1",
)
@click.option("--gamma", type=float, help="masking parameter")
# This is not covered in the paper, but this was for extra experiments
# to see the effect of noises types in masking the image
@click.option("--noise", default="NOISE", type=str, help="Noise to add to the fixation")
@click.option(
    "--output",
    "-o",
    type=click.Path(exists=False),
    default=None,
    help="Output file to save the results64x64, it can be json or csv (if not provided, it will not be saved)",
)
@click.option(
    "--max-number-of-images",
    "-m",
    type=int,
    default=-1,
    help="Maximum number of images to process (for debugging purposes) (default: -1, all images)",
)
# This is for using different datasets
@click.option(
    "--is-new-data",
    "-s",
    is_flag=True,
    default=True,
    help="If the data is new, you can find the new data in `UEyes_dataset` folder"
    "and the old data in `data` folder",
)
@click.option(
    "--images-category",
    "-t",
    type=click.Path(exists=True),
    default=None,
    help="it should be a csv file with two columns: `Image Name` and `Category`",
)
@click.option(
    "--exclude-categories",
    "-e",
    type=str,
    default=None,
    help="it should be a list of categories to exclude, separated by comma",
)
# This is for using different masks for the images (new, old_circle, old_rectangle)
# new is the one used in the paper as new IOR and old_circle is the one used in the paper as old IOR
# old_rectangle is the one that not used in the paper but we used it for extra experiments
@click.option(
    "--mask-type",
    "-k",
    type=str,
    default="new",
    help='can be one of these: "new, "old_circle", "old_rectangle"',
)
def main(
    image_folders,
    csv_folders,
    device,
    number_of_fixations,
    width,
    height,
    radius,
    gamma,
    noise,
    output,
    max_number_of_images,
    is_new_data,
    images_category,
    exclude_categories,
    mask_type,
):
    """
    This script generates fixations for the given images and
    """
    output_extension = None
    if output is not None:
        if os.path.exists(output):
            raise FileExistsError(f"{output} already exists")
        output_extension = output.split(".")[-1]
        if output_extension not in ["json", "csv"]:
            raise ValueError(
                f"Output file should be either json or csv, not {output_extension}"
            )
        click.echo("the results64x64 will be saved in {}".format(output))
    else:
        click.echo("the results64x64 will not be saved")

    if exclude_categories is not None:
        if images_category is None:
            raise ValueError(
                "you can not define the exculded-categories without images-category"
            )
        exclude_categories = exclude_categories.split(",")
    else:
        exclude_categories = []

    df = process_data(csv_folders, is_new_data=is_new_data)
    if images_category is not None:
        images_category = pd.read_csv(images_category, delimiter=";")[
            ["Image Name", "Category"]
        ]
        # check if 'Image Name` is unique
        if len(images_category["Image Name"].unique()) != len(images_category):
            raise ValueError("Image Name column should be unique")

        images_category = images_category.set_index("Image Name")
        # check if all images in df are in images_category
        # if not set(df.index.get_level_values(0)).issubset(set(images_category.index)):
        #     raise ValueError("All images in df should be in images_category")

    device = torch.device(device)
    model = DeepGazeIII(pretrained=True).to(device)
    centerbias = cv2.resize(np.load("centerbias_mit1003.npy"), (width, height))
    centerbias = torch.from_numpy(centerbias).to(device)
    if images_category is not None:
        result = {i: defaultdict(list) for i in images_category["Category"].unique()}
    else:
        result = defaultdict(list)
    for i, (image_address, user_id) in enumerate(tqdm(df.index.unique())):
        if images_category is not None:
            if image_address not in images_category.index:
                continue
            image_category = images_category.loc[image_address]["Category"]
            if image_category in exclude_categories:
                continue
        if i == max_number_of_images:
            break
        image_complete_path = os.path.join(image_folders, image_address)
        if not os.path.exists(image_complete_path):
            continue
        image = cv2.imread(image_complete_path)
        image = cv2.resize(image, (width, height))
        predicted_fixation_points = predict(
            model=model,
            image=image,
            number_of_fixations=number_of_fixations,
            radius=radius,
            device=device,
            image_alter_function=AlternationAlgorithm[noise],
            centerbias_template=centerbias,
            verbose=False,
            mask_type=mask_type,
            gamma=gamma,
        )
        gt_fixation_points = df[df.index == (image_address, user_id)].values
        predicted_fixation_points = np.array(predicted_fixation_points) / np.array(
            [width, height]
        )

        # we have ground truth fixation points and predicted fixation points, so we can calculate
        # the similarity between them

        for func_name, func in EvaluationFunctions.__dict__.items():
            if not func_name.startswith("_") and callable(func.__func__):
                if images_category is not None:
                    image_category = images_category.loc[image_address]["Category"]
                    result[image_category][func_name].append(
                        func(gt_fixation_points, predicted_fixation_points)
                    )
                else:
                    result[func_name].append(
                        func(gt_fixation_points, predicted_fixation_points)
                    )

    if is_new_data:
        results = pd.DataFrame(
            columns=pd.MultiIndex.from_product([list(result.keys()), ["mean", "std"]])
        )
        for category, category_results in result.items():
            results[category, "mean"] = pd.DataFrame(category_results).mean()
            results[category, "std"] = pd.DataFrame(category_results).std()
    else:
        raise NotImplementedError("Old data is not implemented yet")
    if output is None:
        click.echo(results)
    else:
        # for convenience, we save the inputs in the same file as well
        added_values = {
            "width": width,
            "height": height,
            "radius": radius,
            "noise": noise,
            "gamma": gamma,
            "number_of_fixations": number_of_fixations,
            "image_folders": image_folders,
            "csv_folders": csv_folders,
            "device": str(device),
        }
        if output_extension == "json":
            raise NotImplementedError("Saving as json is not implemented yet")

        else:
            for k, v in added_values.items():
                results[k] = v
            save_to_file(results, output)

    return 1


def save_to_file(df: pd.DataFrame, address: str | os.PathLike, **kwargs):
    """
    This functions ensures that we never overwrite a file
    Args:
        df:
        address:
        **kwargs: extra arguments to pass to df.to_csv

    Returns:

    """
    while 1:
        if os.path.exists(address):
            folder, file = os.path.split(address)
            name, ext = os.path.splitext(file)
            address = os.path.join(folder, f"{name}_1{ext}")
        else:
            break
    df.to_csv(address, **kwargs)


@cli.command()
@click.option("--image_folders", "-i", type=click.Path(exists=True))
@click.option("--device", "-d", default="cuda", help="Device to use for training")
@click.option(
    "--number_of_fixations",
    "-n",
    default=10,
    help="Number of fixations to generate, "
    "if zero it will be same as "
    "the number of fixations in GT",
)
@click.option("--width", "-w", type=int, help="Width of the image")
@click.option("--height", "-h", type=int, help="Height of the image")
@click.option(
    "--radius",
    "-r",
    type=float,
    help="Radius of the fixation it should be a float between 0 and 1",
)
@click.option("--gamma", "-g", type=float, help="masking parameter")
@click.option("--noise", type=str, help="Noise to add to the fixation")
@click.option(
    "--output",
    "-o",
    type=click.Path(exists=False),
    default=None,
    help="Output file to save the results64x64, it can be csv",
)
@click.option(
    "--max-number-of-images",
    "-m",
    type=int,
    default=-1,
    help="Maximum number of images to process (for debugging purposes) (default: -1, all images)",
)
@click.option(
    "--mask-type",
    "-k",
    type=str,
    default="new",
    help='can be one of these: "new, "old_circle", "old_rectangle"',
)
def saccade_angle_csv(
    image_folders,
    device,
    number_of_fixations,
    width,
    height,
    radius,
    gamma,
    noise,
    output,
    max_number_of_images,
    mask_type,
):
    """
    This script generates fixations for the given images and saves the results64x64 in a csv file,
    for example, run:
    `python main.py saccade-angle-csv -i ../UEyes_dataset/images -n 10 -w 225 -h 225 -r 0.2 -g 0.5 -o results64x64.csv`
    """
    # check the inputs
    if noise is None:
        noise = "ZERO"
    if os.path.exists(output):
        raise FileExistsError(f"{output} already exists")

    device = torch.device(device)
    model = DeepGazeIII(pretrained=True).to(device)
    centerbias = cv2.resize(np.load("centerbias_mit1003.npy"), (width, height))
    centerbias = torch.from_numpy(centerbias).to(device)

    all_dfs = []
    for index, image_address in enumerate(
        tqdm(glob.glob(os.path.join(image_folders, "*.??g")))
    ):
        if index == max_number_of_images:
            break
        image = cv2.imread(image_address)
        image_height, image_width, _ = image.shape
        fixation_points = predict(
            model=model,
            image=cv2.resize(image, (width, height)),
            number_of_fixations=number_of_fixations,
            radius=radius,
            device=device,
            image_alter_function=AlternationAlgorithm[noise],
            centerbias_template=centerbias,
            verbose=False,
            mask_type=mask_type,
            gamma=gamma,
        )
        # we need to convert the fixation points from the resized image to the original image
        fixation_points = (
            np.array(fixation_points)
            / np.array([width, height])
            * np.array([image_width, image_height])
        )
        df = pd.DataFrame(fixation_points, columns=["x", "y"])
        df["image"] = os.path.basename(image_address)
        df["width"] = width
        df["height"] = height
        df["username"] = "test"
        df["timestamp"] = 0.0
        # I seriously don't have any idea why we need these two last columns :)
        # rearrange the columns
        df = df[["image", "width", "height", "username", "x", "y", "timestamp"]]
        df["radius"] = radius
        df["gamma"] = gamma
        df["mask_type"] = mask_type
        # im putting these columns just for convenience, we may not need them :/
        all_dfs.append(df)
    # As a recall, this function ensures we never overwrite a file
    save_to_file(pd.concat(all_dfs), output, index=False, lineterminator="\n\n")


if __name__ == "__main__":
    cli()
