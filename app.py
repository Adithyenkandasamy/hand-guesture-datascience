import os
import pandas as pd

def rename_folders_from_csv(parent_directory, csv_file):
    # Load CSV file
    df = pd.read_csv(csv_file, header=None)
    
    # Iterate through the CSV file and rename folders
    for index, row in df.iterrows():
        old_name = row[0]  # Assuming the CSV contains only folder names
        old_path = os.path.join(parent_directory, old_name)
        new_name = f"renamed_{old_name}"  # Modify renaming logic as needed
        new_path = os.path.join(parent_directory, new_name)
        
        if os.path.exists(old_path) and os.path.isdir(old_path):
            os.rename(old_path, new_path)
            print(f"Renamed: {old_name} -> {new_name}")
        else:
            print(f"Folder not found: {old_name}")

# Example Usage
parent_directory = "/home/jinwoo/Desktop/hand-guesture-datascience/train/"  # Change this to your actual parent directory
csv_file = "/home/jinwoo/Desktop/hand-guesture-datascience/train.csv"  # Change this to your CSV file path
rename_folders_from_csv(parent_directory, csv_file)