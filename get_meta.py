"""
Extract age and gender from folder structure
"""

import os
import csv

# Directory path
dir_path = r"???/DigitalHandAtlas/DigitalHandAtlas/DICOM/"
output_csv = "image_data.csv"


def extract_image_data_to_csv(dir_path, output_csv):
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        #writer.writerow(["Image Name", "Gender", "Age"])  
        
        for folder in os.listdir(dir_path):
            folder_path = os.path.join(dir_path, folder)
            
            if os.path.isdir(folder_path):
                for subfolder in os.listdir(folder_path):
                    subfolder_path = os.path.join(folder_path, subfolder)
                    
                    if os.path.isdir(subfolder_path):
                        race = subfolder[:3]  # Extract race (first 3 characters)
                        gender = subfolder[3:4]  # Extract gender (3rd character)
                        age_2 = subfolder[4:]  # Extract age (rest of the characters)

                        if int(age_2) <= 13:
                            age = "Young"
                        else:
                            age = "Old"
                        
                        for file_name in os.listdir(subfolder_path):
                            if file_name.endswith(".dcm") or file_name.endswith(".DCM"):
                                # Extract image name (without extension)
                                image_name = os.path.splitext(file_name)[0]
                                
                                # Write the data to the CSV
                                writer.writerow([int(image_name), gender, age, int(age_2)])


extract_image_data_to_csv(dir_path, output_csv)

