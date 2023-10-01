import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import os
import pandas as pd
from sklearn.neighbors import KDTree
import pickle
import argparse
from data.streetlearn_pickle import TrainingTuple


def construct_query_dict(query_filepath, output_filepath, ind_nn_r, ind_r_r=50):
    # ind_nn_r: threshold for positive examples
    # ind_r_r: threshold for negative examples
    db = pd.read_csv(query_filepath, sep=',', header=None).values
    x = np.expand_dims(pd.to_numeric(db[:,1]),axis=1)
    y = np.expand_dims(pd.to_numeric(db[:,2]),axis=1)
    locations = np.concatenate((x,y),-1)

    tree = KDTree(locations)
    ind_nn = tree.query_radius(locations, r=ind_nn_r)
    ind_r = tree.query_radius(locations, r=ind_r_r)
    queries = {}
    for ndx in range(len(ind_nn)):
        panoid = db[ndx][0]
        center_x = db[ndx][1]
        center_y = db[ndx][2]
        heading = db[ndx][3]
        city = db[ndx][4]
        positives = ind_nn[ndx]
        non_negatives = ind_r[ndx]
        # positives = positives[positives != ndx]  # comment this line: postives including itself
        # Sort ascending order
        positives = np.sort(positives)
        non_negatives = np.sort(non_negatives)
        queries[ndx] = TrainingTuple(id=panoid, center_x=center_x, center_y=center_y, heading=heading,
                                city=city, positives=positives, non_negatives=non_negatives)

    with open(output_filepath, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done ", output_filepath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Baseline training dataset')
    parser.add_argument('--dataset_root', type=str, default='datasets', required=False, help='Dataset root folder')
    parser.add_argument('--train_file', type=str, default='wallstreet5kU', required=False, help='Train file')

    args = parser.parse_args()
    print('Dataset root: {}'.format(args.dataset_root))
    assert os.path.exists(args.dataset_root), f"Cannot access dataset root folder: {args.dataset_root}"

    query_filepath = os.path.join(args.dataset_root, 'csv', args.train_file+'_xy.csv')  
    output_filepath = os.path.join(args.dataset_root, 'csv', args.train_file+'.pickle')

    # ind_nn_r is a threshold for positive elements - 10 is in original PointNetVLAD code for refined dataset
    construct_query_dict(query_filepath, output_filepath, ind_nn_r=10, ind_r_r=50)
