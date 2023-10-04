import os
import sys
sys.path.append(os.getcwd())
import csv

# Define a function to find neighbors for a given node
def find_neighbors(node, yaw, relationships):
    if node in relationships:
        neighbors = []
        bearings = []
        for i, tp in enumerate(relationships[node]):
            neighbor = tp[0]
            bearing = tp[1]
            angle_difference = abs(yaw - bearing)
            angle_difference = min(angle_difference, 360 - angle_difference)
            if angle_difference < 90:
                neighbors.append(neighbor)
                bearings.append(bearing)
        return neighbors, bearings
    else:
        return [], []


if __name__ == '__main__':
    # Read the links.txt file and create the relationships dictionary
    relationships = {}
    city = 'manhattan'
    txt_file = os.path.join('datasets','jpegs_'+city+'_2019','links.txt')
    with open(txt_file, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            node1, bearing, node2 = parts
            bearing = int(bearing)
            if node1 in relationships:
                relationships[node1].append((node2,bearing))
            else:
                relationships[node1] = [(node2,bearing)]

    # Read the CSV file
    area = 'unionsquare5kU'
    csv_file = os.path.join('datasets','csv',area+'_xy.csv')
    data = []
    local_ids = []
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            data.append(row)
            local_ids.append(row[0])

    # Add a new column 'Neighbors' to the CSV data
    for i, row in enumerate(data):
        node = row[0]
        yaw = float(row[3])
        neighbors, bearings = find_neighbors(node, yaw, relationships)
        if len(neighbors) > 0:
            position = []
            bear = []
            for i, neighbor in enumerate(neighbors):
                try:
                    position.append(local_ids.index(neighbor))
                    bear.append(bearings[i])
                except ValueError:
                    print(f"'{neighbor}' is not in local_list.")
            row.append(position)
            row.append(bear)
        else:
            row.append([])
            row.append([])

    # Write the updated data back to a new CSV file
    output_file = os.path.join('datasets','csv',area+'_nb.csv')
    with open(output_file, 'w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerows(data)

    print("Neighbors have been added to the CSV file (_nb.csv).")

