import csv
import os

input_dir = "/home/ava/projects/trajectory-model/data/mocap_new/big/full/spill-free"
output_dir = "/home/ava/projects/trajectory-model/data/mocap_new/big/full/spill-free/filtered"

input_dir_files = os.listdir(input_dir)

for file in input_dir_files:
    if file == "filtered":
        continue
    input_file_path = os.path.join(input_dir, file)
    output_file_path = os.path.join(output_dir, file)

    with open(input_file_path, mode='r') as input_file, open(output_file_path, mode='w', newline='') as output_file:
        csv_reader = csv.reader(input_file)
        csv_writer = csv.writer(output_file)
        for row in csv_reader:
            row_values = [float(value) for value in row[1:]]
            if row_values == [-0.0, -0.0, 0.0, -0.0, -0.0, 0.0, -1.0]:
                continue 
            csv_writer.writerow(row)            

    input_file.close()
    output_file.close()