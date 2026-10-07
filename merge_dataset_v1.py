import os
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

DATASET_SUFFIX = "_Dataset"
MERGED_FOLDER_NAME = "Merged_Dataset"


# ============================================================
# MERGE ONE USER'S DATASET
# ============================================================

def merge_one_user(user_folder, merged_folder):

    # --------------------------------------------------------
    # Get username
    #
    # Example:
    # Abhirup_Dataset -> Abhirup
    # --------------------------------------------------------

    folder_name = os.path.basename(
        os.path.normpath(user_folder)
    )

    username = folder_name[:-len(DATASET_SUFFIX)]

    print("\n" + "=" * 70)
    print(f"Processing user: {username}")
    print("=" * 70)

    # --------------------------------------------------------
    # Find CSV files ONLY inside this user's folder
    # --------------------------------------------------------

    csv_files = []

    for filename in os.listdir(user_folder):

        file_path = os.path.join(
            user_folder,
            filename
        )

        if (
            os.path.isfile(file_path)
            and filename.lower().endswith(".csv")
        ):
            csv_files.append(file_path)

    # Sort alphabetically
    csv_files.sort()

    if not csv_files:

        print("No CSV files found.")
        return

    print(f"CSV files found: {len(csv_files)}")

    # --------------------------------------------------------
    # Store this user's alphabet data
    # --------------------------------------------------------

    user_data = []

    # --------------------------------------------------------
    # Read each alphabet CSV
    # --------------------------------------------------------

    for csv_file in csv_files:

        filename = os.path.basename(csv_file)

        # ----------------------------------------------------
        # Filename becomes label
        #
        # A.csv -> A
        # B.csv -> B
        # ...
        # Z.csv -> Z
        # ----------------------------------------------------

        label = os.path.splitext(filename)[0].upper()

        print(
            f"Reading {filename} -> Label: {label}"
        )

        try:

            df = pd.read_csv(csv_file)

        except Exception as e:

            print(
                f"ERROR reading {filename}: {e}"
            )

            continue

        # ----------------------------------------------------
        # Skip empty CSV
        # ----------------------------------------------------

        if df.empty:

            print(
                f"WARNING: {filename} is empty. Skipping."
            )

            continue

        # ----------------------------------------------------
        # Remove Timestamp
        # ----------------------------------------------------

        if "Timestamp" in df.columns:

            df.drop(
                columns=["Timestamp"],
                inplace=True
            )

        # ----------------------------------------------------
        # Remove label if it already exists
        # ----------------------------------------------------

        if "label" in df.columns:

            df.drop(
                columns=["label"],
                inplace=True
            )

        # ----------------------------------------------------
        # Add label at the END
        # ----------------------------------------------------

        df["label"] = label

        # ----------------------------------------------------
        # Add ONLY this alphabet's data
        # to this user's list
        # ----------------------------------------------------

        user_data.append(df)

        print(
            f"   Rows: {len(df)}"
        )

    # --------------------------------------------------------
    # Check whether data exists
    # --------------------------------------------------------

    if not user_data:

        print(
            f"No valid CSV data found for {username}."
        )

        return

    # --------------------------------------------------------
    # Merge THIS USER'S 26 alphabet files
    #
    # IMPORTANT:
    # This happens separately for every user.
    # --------------------------------------------------------

    merged_df = pd.concat(
        user_data,
        ignore_index=True
    )

    # --------------------------------------------------------
    # Output filename
    # --------------------------------------------------------

    output_filename = (
        f"{username}_merged_dataset.csv"
    )

    output_path = os.path.join(
        merged_folder,
        output_filename
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    merged_df.to_csv(
        output_path,
        index=False
    )

    # --------------------------------------------------------
    # Show result
    # --------------------------------------------------------

    print("\nMERGE COMPLETED")

    print(f"User         : {username}")
    print(f"CSV files    : {len(csv_files)}")
    print(f"Total rows   : {len(merged_df)}")
    print(f"Total columns: {len(merged_df.columns)}")

    print(
        f"Saved to     : {output_path}"
    )

    # --------------------------------------------------------
    # Show label distribution
    # --------------------------------------------------------

    print("\nLabel distribution:")

    label_counts = (
        merged_df["label"]
        .value_counts()
        .sort_index()
    )

    for label, count in label_counts.items():

        print(
            f"    {label} : {count}"
        )


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("=" * 70)
    print("ISL DATASET MERGER")
    print("=" * 70)

    # --------------------------------------------------------
    # Ask for the MAIN Dataset folder
    # --------------------------------------------------------

    dataset_folder = input(
        "\nEnter the Dataset folder path: "
    ).strip()

    # Remove quotes if path was pasted with quotes
    dataset_folder = dataset_folder.strip('"').strip("'")

    # --------------------------------------------------------
    # Check Dataset folder
    # --------------------------------------------------------

    if not os.path.isdir(dataset_folder):

        print(
            "\nERROR: Dataset folder does not exist."
        )

        return

    # --------------------------------------------------------
    # Get parent folder
    #
    # Example:
    #
    # C:\ISL\Project\Dataset
    #
    # parent:
    #
    # C:\ISL\Project
    # --------------------------------------------------------

    dataset_folder = os.path.abspath(
        dataset_folder
    )

    parent_folder = os.path.dirname(
        dataset_folder
    )

    # --------------------------------------------------------
    # Create Merged_Dataset OUTSIDE Dataset
    # --------------------------------------------------------

    merged_folder = os.path.join(
        parent_folder,
        MERGED_FOLDER_NAME
    )

    os.makedirs(
        merged_folder,
        exist_ok=True
    )

    print(
        f"\nMerged folder:"
    )

    print(
        merged_folder
    )

    # --------------------------------------------------------
    # Find ONLY the immediate user folders
    #
    # We do NOT search recursively.
    #
    # Therefore:
    #
    # Abhirup_Dataset -> one user
    # Adipsha_Dataset -> one user
    # etc.
    # --------------------------------------------------------

    user_folders = []

    for item in sorted(
        os.listdir(dataset_folder)
    ):

        item_path = os.path.join(
            dataset_folder,
            item
        )

        if (
            os.path.isdir(item_path)
            and item.endswith(DATASET_SUFFIX)
        ):

            user_folders.append(item_path)

    # --------------------------------------------------------
    # Check users
    # --------------------------------------------------------

    if not user_folders:

        print(
            "\nERROR: No *_Dataset folders found."
        )

        return

    # --------------------------------------------------------
    # Display users
    # --------------------------------------------------------

    print(
        f"\nUsers found: {len(user_folders)}"
    )

    for folder in user_folders:

        print(
            f"    {os.path.basename(folder)}"
        )

    # --------------------------------------------------------
    # Process EACH user separately
    # --------------------------------------------------------

    for user_folder in user_folders:

        merge_one_user(
            user_folder,
            merged_folder
        )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print("\n\n" + "=" * 70)
    print("ALL USERS PROCESSED")
    print("=" * 70)

    print(
        f"\nMerged files are available in:"
    )

    print(
        merged_folder
    )

    print("\nCreated files:")

    for filename in sorted(
        os.listdir(merged_folder)
    ):

        if filename.lower().endswith(".csv"):

            print(
                f"    {filename}"
            )

    print("=" * 70)


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()