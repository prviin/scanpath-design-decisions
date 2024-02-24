import os
import random
from enum import Enum
from typing import Callable

import cv2
import numpy as np
import torch

import deepgaze_pytorch
from data import process_data


class AlternationAlgorithm(Enum):
    BLUR = "blur"
    NOISE = "noise"
    ZERO = "zero"
    SALT_AND_PEPPER = "salt_and_pepper"
    PIXEL_PERMUTATION = "pixel_permutation"
    # This is not used in paper, but it's added for more experiments


def salt_and_pepper(
    image: np.ndarray, prob: float = 0.05, salt: float = 0.5
) -> np.ndarray:
    """

    :param image:
    :param prob: the probability of the noise
    :param salt: the probability of the salt
    :return:
    """
    noise_mask = np.random.choice(
        [0, 1, 2], size=image.shape[:2], p=[1 - prob, prob * salt, prob * (1 - salt)]
    )
    # 0 for no noise, 1 for salt, 2 for pepper
    image = image.copy()
    image[noise_mask == 1, :] = 255
    image[noise_mask == 2, :] = 0
    return image


def pixel_permutation(image: np.ndarray, kernel_size: int):
    """
    this function will tile the image into small patches and permute the pixels in each patch\

    :param image: the image to be processed
    :param kernel_size: the size of the patch
    :return:
    """
    image = image.copy()
    for i in range(0, image.shape[0], kernel_size):
        for j in range(0, image.shape[1], kernel_size):
            image[i : i + kernel_size, j : j + kernel_size] = permute_kernel(
                image[i : i + kernel_size, j : j + kernel_size]
            )
    return image


def permute_kernel(patch: np.ndarray):
    """
    this function will permute the pixels in a patch
    :param patch:
    :return:
    """
    x_indices = np.arange(patch.shape[1])
    y_indices = np.arange(patch.shape[0])
    np.random.shuffle(x_indices)
    np.random.shuffle(y_indices)
    return patch[y_indices, :][:, x_indices]


def predict(
    model: torch.nn.Module,
    image: np.ndarray,
    number_of_fixations: int,
    radius: tuple[int, int] | float,
    gamma: float,
    image_alter_function: Callable | AlternationAlgorithm | np.ndarray,
    device: torch.device,
    centerbias_template: torch.Tensor = None,
    initial_fixation_points=None,
    verbose=False,
    mask_type: str = "new",
) -> list:
    """
    :param model: model to be used for prediction
    :param image: image to be predicted
    :param number_of_fixations: number of fixations to be made
    :param radius: radius of the IOR circle
    :param gamma: gamma value to be used for masking parameter
    :param image_alter_function: function to be used for image alteration, if numpy.ndarray is given, it is used as
                                 altered image
    :param device: device to be used for prediction
    :param centerbias_template: centerbias template to be used for prediction if None, uniform centerbias is used
    :param initial_fixation_points: initial fixation points to be used for prediction, if None, the center of the image
                                    is used
    :param verbose: if True, the progress of the prediction will be printed
    :param mask_type: can be one of these: "new, "old_circle", "old_rectangle"
    :return: List of predicted fixations [[x0, y0], [x1, y1], ...]
    """
    torch.cuda.empty_cache()
    if centerbias_template is None:
        centerbias_template = torch.zeros(image.shape)

    if image.shape[:2] != centerbias_template.shape[:2]:
        raise ValueError(
            f"Image and centerbias template must have the same shape, "
            f"got {image.shape} and {centerbias_template.shape}"
        )

    if isinstance(radius, float):
        if radius > 1 or radius < 0:
            raise ValueError(f"Radius must be between 0 and 1, got {radius}")
        r0 = int(radius * image.shape[0])
        r1 = int(radius * image.shape[1])
        radius = (r0, r1)
        # it's the radius of the IOR rectangle in mask
    altered_image = np.array([])
    if isinstance(image_alter_function, AlternationAlgorithm):
        if image_alter_function not in AlternationAlgorithm:
            raise ValueError(
                f"Image alter function must be one of {AlternationAlgorithm.__members__}, "
                f"got {image_alter_function}"
            )

        if image_alter_function == AlternationAlgorithm.BLUR:
            altered_image = cv2.GaussianBlur(image, (901, 901), 50)
        elif image_alter_function == AlternationAlgorithm.NOISE:
            altered_image = image + np.random.normal(100, 15, image.shape)
        elif image_alter_function == AlternationAlgorithm.ZERO:
            altered_image = np.zeros_like(image)
        elif image_alter_function == AlternationAlgorithm.SALT_AND_PEPPER:
            altered_image = salt_and_pepper(image, prob=0.5, salt=0.5)
        elif image_alter_function == AlternationAlgorithm.PIXEL_PERMUTATION:
            altered_image = pixel_permutation(image, 50)
    elif callable(image_alter_function):
        altered_image = image_alter_function(image)

    elif isinstance(image_alter_function, np.ndarray):
        altered_image = image_alter_function
    else:
        raise ValueError(
            "image altered function must be one of the following: "
            "AlternationAlgorithm instance, callable or numpy.ndarray"
        )

    if altered_image.shape != image.shape:
        raise ValueError(
            "altered image and the original image must have the same shape, "
            f"got {altered_image.shape} and {image.shape}"
        )
    image = torch.from_numpy(image).to(device)
    altered_image = torch.from_numpy(altered_image).type(torch.float32).to(device)
    centerbias = centerbias_template - torch.logsumexp(centerbias_template, dim=(0, 1))
    if initial_fixation_points is None:
        fixation_points = [[image.shape[0] // 2, image.shape[1] // 2]] * 4
        # deep gaze 3 model only accepts 4 fixation points
    else:
        fixation_points = initial_fixation_points

    assert len(fixation_points) == 4, (
        "fixation points must have 4 points, but got "
        f"{len(fixation_points)} points\n {fixation_points=}"
    )
    assert all(
        len(i) == 2 for i in fixation_points
    ), f"fixation points must have two columns, but got {fixation_points}"

    # Fixation points will be returned as a list of [x, y] coordinates
    mask: torch.Tensor = torch.zeros_like(image, dtype=torch.float).to(device)
    for i in range(number_of_fixations):
        if verbose:
            print(
                f"Fixation {i + 1}/{number_of_fixations}, founded {len(fixation_points)} fixation points"
            )

        mask = create_mask(mask, fixation_points, radius, gamma, mask_type, device)

        new_image = image * (1 - mask) + altered_image * mask
        new_image = new_image.permute(2, 0, 1).unsqueeze(0)
        # TODO: this line may be time consuming, we may need to change it
        if verbose:
            cv2.imwrite(f"mask{i}.png", mask.detach().cpu().numpy() * 255)
            cv2.imwrite(
                f"new_image{i}.png",
                new_image.squeeze().permute(1, 2, 0).detach().cpu().numpy(),
            )
            cv2.imwrite(f"blured_image{i}.png", altered_image.detach().cpu().numpy())

        fixation_points_tensor = torch.tensor(np.array(fixation_points[-4:])).to(
            image.device
        )
        # since the deep gaze 3 model only accepts 4 fixation points, we use the last 4 fixation points
        assert fixation_points_tensor.shape[1] == 2, (
            "fixation points must have two columns, but got "
            f"{fixation_points_tensor.shape[1]} columns"
        )
        # Fixation points have two columns, the first column is x, and the second column is y
        model_output = model(
            new_image,
            centerbias.unsqueeze(0),
            fixation_points_tensor[:, 0].unsqueeze(0),
            fixation_points_tensor[:, 1].unsqueeze(0),
        )
        assert torch.all(
            mask[:, :, 0] == mask[:, :, 2]
        ), "mask must be a grayscale image"
        mask_ = mask[:, :, 0]

        log_density_prediction = (100 + model_output.squeeze()) * (
            1 - mask_
        ) - mask_ * 1000

        # Find the brightest pixel in the probaility map
        brightest_pixel = (
            (log_density_prediction == torch.max(log_density_prediction))
            .nonzero()[0]
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )

        fixation_points.append(brightest_pixel)
    return fixation_points


def create_mask(
    mask: torch.Tensor,
    fixation_points: list,
    radius: tuple[int, int],
    gamma: float,
    type: str,
    device: torch.device,
) -> torch.Tensor:
    """
    this function will create propper arguments and will call the proper function to create the mask
    Args:
        device:
        mask: is the previous mask
        fixation_points: is a list of fixation points [[x0, y0], [x1, y1], ...]
        radius: is a tuple of radius of the IOR circle (r0, r1)
        gamma: is the gamma value to be used for masking parameter
        type: can be one of these: "new, "old_circle", "old_rectangle"
        device: device to be used for prediction, if None, the default device will be used (it will be used only if the
                type is "old_circle" or "old_rectangle")

    Returns:
        mask: the new mask (torch.Tensor)

    """
    if device is None:
        device = torch.device("cuda")
    if type == "new":
        return create_mask_new(mask, fixation_points, radius[0], radius[1], gamma).to(
            device
        )
    elif type == "old_circle":
        return create_mask_old_circular(
            mask.shape[0], mask.shape[1], fixation_points, min(radius), gamma
        ).to(device)
    elif type == "old_rectangle":
        return create_mask_old_rectangle(
            torch.zeros_like(mask), fixation_points, radius[0], radius[1], gamma
        ).to(device)
    elif type == "new_circle":
        return create_mask_new_circle(
            mask.shape[0], mask.shape[1], fixation_points, min(radius), gamma
        ).to(device)
    else:
        raise ValueError(
            f"not valid mask type, must be one of these: 'new', 'old_circle', 'old_rectangle' not {type}"
        )


def create_mask_new(
    mask: torch.Tensor, fixations_point: list, r0: int, r1: int, gamma: float
) -> torch.Tensor:
    """
    this function creates a mask with the given center and radius (we use rectangle instead of ellipse for simplicity)
    initially this function will multiply the mask with 0.9 and then will create a rectangle with the given center
    and radius with value of 1
    """
    mask = torch.zeros_like(mask)
    for fi, (x, y) in enumerate(fixations_point):
        c = gamma ** (len(fixations_point) - fi - 1)
        mask[x - r0 : x + r0, y - r1 : y + r1] = c
    return mask


def create_mask_old_circular(
    h: int,
    w: int,
    fixations_point: list[list[int, int], ...],
    radius: int,
    gamma: float,
):
    # get the circular mask
    mask = torch.zeros(h, w)
    Y, X = np.ogrid[:h, :w]
    for i, (x, y) in enumerate(fixations_point):
        dist = np.sqrt((X - x) ** 2 + (Y - y) ** 2)
        c = 1 - gamma * (len(fixations_point) - i - 1)
        mask = torch.maximum(mask, torch.from_numpy(dist <= radius) * c)
    return torch.unsqueeze(mask, 2).repeat(1, 1, 3)


def create_mask_old_rectangle(
    mask: torch.Tensor,
    fixations_point: list[list[int, int], ...],
    r0: int,
    r1: int,
    gamma: float,
) -> torch.Tensor:
    # get the rectangle mask
    len_fixations = len(fixations_point)
    for fi, (x, y) in enumerate(fixations_point):
        c = 1 - gamma * (len_fixations - fi - 1)
        mask[x - r0 : x + r0, y - r1 : y + r1] = c

    return mask


def create_mask_new_circle(
    h: int,
    w: int,
    fixations_point: list[list[int, int], ...],
    radius: int,
    gamma: float,
) -> torch.Tensor:
    # get the circular mask
    mask = torch.zeros(h, w)
    X, Y = np.ogrid[:h, :w]
    for i, (x, y) in enumerate(fixations_point):
        c = gamma ** (len(fixations_point) - i - 1)
        dist = np.sqrt((X - x) ** 2 + (Y - y) ** 2)
        mask = torch.maximum(mask, torch.from_numpy(dist <= radius) * c)
    return torch.unsqueeze(mask, 2).repeat(1, 1, 3)


def draw_fixation_points(
    image_address: str | os.PathLike,
    image_size: (int, int),
    model: torch.nn.Module,
    number_of_fixations: int,
    radius: tuple[int, int] | float,
    gamma: float,
    image_alter_function: Callable | np.ndarray | AlternationAlgorithm,
    device: torch.device,
    csv_folders,
    centerbias_template: np.ndarray = None,
    mask_type: str = "new_circle",
):
    image = cv2.imread(image_address)
    if image_size[0] < 0 or image_size[1] < 0:
        image_size = image.shape[:2][::-1]
        print("image size", image_size)
    resized_image = cv2.resize(image, image_size)
    print(f"{image.shape}=image_size,{resized_image.shape}=resized_image")
    df = process_data(csv_folders, is_new_data=True)
    image_name = os.path.basename(image_address)
    available_users = df.loc[image_name].index.unique()

    random_user = random.choice(available_users)
    gt_fixation_points = (
        df.loc[(image_name, random_user)].values
        * np.array([image.shape[1], image.shape[0]])
    )[:number_of_fixations, :]
    print(gt_fixation_points)
    if resized_image.shape[:2] != centerbias_template.shape[:2]:
        centerbias_template = cv2.resize(
            centerbias_template, (resized_image.shape[1], resized_image.shape[0])
        )
    centerbias_template = torch.from_numpy(centerbias_template).to(device)
    coef = np.array(
        [
            image.shape[1] / resized_image.shape[1],
            image.shape[0] / resized_image.shape[0],
        ]
    )

    results = predict(
        model=model,
        image=resized_image,
        number_of_fixations=number_of_fixations,
        radius=radius,
        gamma=gamma,
        image_alter_function=image_alter_function,
        device=device,
        centerbias_template=centerbias_template,
        mask_type=mask_type,
        verbose=True,
    )
    results = np.array(results)[:, :]
    print(coef)

    results = results * coef
    results = results.astype(int)

    line_color = (0, 0, 0)

    print(gt_fixation_points)

    draw_point(
        image * 0.6 + 90,
        gt_fixation_points.astype(int),
        line_color,
        output_name=f"gt_{image_name.split('.')[0]}",
    )
    draw_point(
        image * 0.6 + 90,
        results,
        line_color,
        output_name=f"prdicted_{image_name.split('.')[0]}",
    )


def draw_point(
    image: np.ndarray, points: np.ndarray, line_color: (int, int, int), output_name: str
):
    c = 0
    color_step = 255 // points.shape[0]
    color = [0, 0, 255]
    radius = 15
    image = image.copy()
    for x, y in points:
        color[0] += color_step
        color[2] -= color_step
        cv2.circle(image, (x, y), radius, color, -1)
        if c != 0:
            cv2.line(image, (points[c - 1][0], points[c - 1][1]), (x, y), color, 2)
        c += 1
    cv2.imwrite(f"visualization/{output_name}.png", image)
    print(points)
    print("#################################")


if __name__ == "__main__":
    device = torch.device("cuda")
    model = deepgaze_pytorch.DeepGazeIII(pretrained=True).to(device)
    # if you are calling predict directly use this line for image
    # image = cv2.imread("../UEyes_dataset/images/a3fb58.jpg")
    # otherwise, if you are calling draw_fixation_points function, use this line:
    image = "../UEyes_dataset/images/20e9c6.jpg"

    # image = torch.tensor(image).to(device)
    centerbias = np.load("centerbias_mit1003.npy")
    # if you are calling predict directly use this line for centerbias
    # centerbias = cv2.resize(centerbias, (image.shape[1], image.shape[0]))
    # centerbias = torch.from_numpy(centerbias).to(device)
    # otherwise, if you are using draw_fixation_points function, use none of them :)

    # res = predict(model=model, image=image, number_of_fixations=10, radius=.05, device=device,
    #               image_alter_function=AlternationAlgorithm.PIXEL_PERMUTATION, centerbias_template=centerbias,
    #               verbose=True, gamma=0.9)
    draw_fixation_points(
        image,
        (225, 225),
        model,
        number_of_fixations=10,
        radius=0.2,
        gamma=0.1,
        image_alter_function=AlternationAlgorithm.ZERO,
        device=device,
        centerbias_template=centerbias,
        mask_type="new_circle",
        csv_folders="../UEyes_dataset/eyetracker_logs",
    )
