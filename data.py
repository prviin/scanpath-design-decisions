import glob
import os
import re
import pandas as pd


def process_csv(file_name, calculate_fixation_duration: bool, minimum_number_of_fixations: int, is_new_data) -> pd.DataFrame:
    """

    Args:
        file_name: address of the csv file
        calculate_fixation_duration: not implemented yet
        minimum_number_of_fixations: the fixations with less than this number will be removed
        is_new_data: you can find the structure of the new data in the `UEyes_dataset` folder and the old data in the
                     `data` folder

    Returns:

    """
    block_number, user_id = re.findall(r"(\d\d)_KH(\d\d\d)_fixations\.csv", file_name,
                                       flags=re.IGNORECASE)[0]
    df_ = pd.read_csv(file_name)
    df = df_[["MEDIA_NAME", "BPOGX", "BPOGY", "BPOGV"]]
    if calculate_fixation_duration:
        # calculate the duration of each fixation
        # we should read column number 4
        # and calculate the difference between each row
        raise NotImplementedError("This feature is not implemented yet")
    df = df[df['BPOGV'] == 1]
    df = df[~ ((df['BPOGX'] == 0.5) & (df['BPOGY'] == 0.5))]
    df = df[(df['BPOGX'] > 0) & (df['BPOGY'] > 0) & (df['BPOGX'] < 1) & (df['BPOGY'] < 1)]
    if is_new_data:
        df["image_name"] = df["MEDIA_NAME"]
    else:
        df["image_name"] = f"block {block_number}/" + df["MEDIA_NAME"]
    df["user_id"] = user_id
    df.drop(columns=["MEDIA_NAME", "BPOGV"], inplace=True)
    df.set_index(['image_name', 'user_id'], inplace=True)
    df.rename(columns={'BPOGX': 'x', 'BPOGY': 'y'}, inplace=True)
    df = df.groupby(['image_name', 'user_id']).filter(lambda x: len(x) >= minimum_number_of_fixations)
    return df


def process_data(csv_folders: str | os.PathLike, calculate_fixation_duration: bool = False,
                 minimum_number_of_fixations: int = 3, is_new_data=True) -> pd.DataFrame:
    r"""

    Args:
        calculate_fixation_duration: if True, the duration of each fixation will be calculated
                                     and added to the dataframe
        csv_folders: This folder should contain the folders with the csv files,
                     each folder should be named with the  "kh\d\d\d" format
                     csv files should be named with the "\d\d_kh\d\d\d_fixations.csv" format
        minimum_number_of_fixations: the minimum number of fixations for each image
        is_new_data: you can find the structure of the new data in the `UEyes_dataset` folder and the old data in the
                     `data` folder

    Returns: A dataframe with the data from all the csv files, the indexes are:
             - image_name: the name of the image
             - user_id: the id of the user
             and the columns are:
             - x: the x coordinate of the fixation
             - y: the y coordinate of the fixation
             - timestamp: the timestamp of the fixation (if calculate_fixation_duration is True)
    """
    all_dataframes = []
    if is_new_data:
        data_folders = os.path.join(csv_folders, "*.csv")
    else:
        data_folders = os.path.join(csv_folders, "*", "*.csv")

    for file_name in glob.glob(data_folders):
        all_dataframes.append(process_csv(file_name, calculate_fixation_duration,
                                          minimum_number_of_fixations, is_new_data))
    if len(all_dataframes) == 0:
        raise FileNotFoundError("No csv files found")
    all_dataframes = pd.concat(all_dataframes)
    return all_dataframes


if __name__ == '__main__':
    df = process_data("../UEyes_dataset/eyetracker_logs", is_new_data=True)
    print(df)

    print("average number of fixation points:",df.groupby(df.index).count()['x'].mean())
    print(df.groupby(df.index).count()['x'].std())
