import pandas as pd

"""
Generates filtered train and test CSV files from a given dataset.
This function reads a CSV file containing dataset information and two text files
containing lists of filenames for training/validation and testing. It filters the
rows in the dataset based on these filenames and saves the filtered data into
separate CSV files for training and testing.
Files:
    - datasets/data/nih/Data_Entry_2017_v2020.csv: The input CSV file containing dataset information.
    - datasets/data/nih/train_val_list.txt: A text file containing filenames for training/validation.
    - datasets/data/nih/test_list.txt: A text file containing filenames for testing.
    - datasets/data/nih/train.csv: The output CSV file for filtered training data.
    - datasets/data/nih/test.csv: The output CSV file for filtered testing data.
Returns:
    None
Prints:
    A message indicating the locations of the saved filtered data files.
"""


def genarate_csv():
    csv_file = "datasets/data/nih/Data_Entry_2017_v2020.csv"
    train_val_file = "datasets/data/nih/train_val_list.txt"
    test_file = "datasets/data/nih/test_list.txt"
    output_train_csv = "datasets/data/nih/train.csv"
    output_test_csv = "datasets/data/nih/test.csv"

    with open(train_val_file, "r") as f:
        train_filenames = set(f.read().splitlines())

    with open(test_file, "r") as f:
        test_filenames = set(f.read().splitlines())

    df = pd.read_csv(csv_file)

    image_column = "Image Index"
    label_column = "Finding Labels"

    # filter out the rows with invalid filenames
    df_train_filtered = df[df[image_column].isin(train_filenames)]
    df_test_filtered = df[df[image_column].isin(test_filenames)]

    df_train_labels = df_train_filtered[label_column].str.get_dummies(sep="|")
    df_test_labels = df_test_filtered[label_column].str.get_dummies(sep="|")

    df_train_filtered = pd.concat(
        [df_train_filtered[[image_column]], df_train_labels], axis=1
    )
    df_test_filtered = pd.concat(
        [df_test_filtered[[image_column]], df_test_labels], axis=1
    )

    # remove the "No Finding" column
    df_train_filtered = df_train_filtered.drop(columns=["No Finding"])
    df_test_filtered = df_test_filtered.drop(columns=["No Finding"])

    df_train_filtered.to_csv(output_train_csv, index=False)
    df_test_filtered.to_csv(output_test_csv, index=False)

    print(f"Filtered data saved to {output_train_csv} and {output_test_csv}")


if __name__ == "__main__":
    genarate_csv()
