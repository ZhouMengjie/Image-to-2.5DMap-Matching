import os
import ast
import sys
sys.path.append(os.getcwd())
import csv
import numpy as np

def build_sequences(sequences, data):
    seq_num = len(sequences)
    new_sequences = []
    for i in range(seq_num):
        idx = sequences[i,-1]
        neighbors = data[idx][5]
        neighbors = ast.literal_eval(neighbors)
        
        if not neighbors:
            continue
        
        for j in range(len(neighbors)):
            k = np.where(sequences[i, :] == neighbors[j])[0]
            if len(k) == 0:
                new_sequences.append(list(np.concatenate([sequences[i, :],[neighbors[j]]])))
    new_sequences = np.array(new_sequences)
    return new_sequences


if __name__ == '__main__':
    # Read the CSV file
    area = 'hudsonriver5kU'
    csv_file = os.path.join('datasets','csv',area+'_nb.csv')
    data = []
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            data.append(row)

    # Generate sequences to the CSV data
    max_seq_length = 5
    sequences = np.arange(len(data))
    sequences = sequences.reshape((len(data), 1))
    i = 1
    while i < max_seq_length:
        sequences = build_sequences(sequences, data)
        i += 1 

    # If required, randomly shuffle each sequence
    # Set a seed for reproducibility
    # np.random.seed(0)
    # Shuffle each row independently
    # np.apply_along_axis(np.random.shuffle, axis=1, arr=sequences)

    # Write the updated data back to a new CSV file
    output_file = os.path.join('datasets','csv',area+'_sq.csv')
    with open(output_file, 'w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerows(sequences)

    print("Sequences have been added to the CSV file (_sq.csv).")

