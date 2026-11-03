
import pandas as pd
from pathlib import Path

def merge_csv_files():
    folder_path = input("Enter the folder path containing the 5 CSV files: ").strip()
    folder = Path(folder_path.strip('"'))

    if not folder.is_dir():
        print("Error: Folder not found.")
        return

    filenames = ["A.csv", "B.csv", "G.csv", "T.csv", "X.csv"]
    dataframes = []

    for filename in filenames:
        file_path = folder / filename

        if not file_path.is_file():
            print(f"Error: Missing file: {filename}")
            return

        try:
            df = pd.read_csv(file_path)
            df["Alphabet"] = filename[0].upper()
            dataframes.append(df)
            print(f"Loaded {filename}: {len(df)} rows")
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            return

    merged_df = pd.concat(dataframes, ignore_index=True)

    output_path = folder / "merged_dataset.csv"
    merged_df.to_csv(output_path, index=False)

    print("\nMerge completed successfully.")
    print(f"Total rows: {len(merged_df)}")
    print(f"Output file: {output_path}")

if __name__ == "__main__":
    merge_csv_files()
