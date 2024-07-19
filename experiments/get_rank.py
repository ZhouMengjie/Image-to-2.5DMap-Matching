import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
plt.rc('font',family='Times New Roman') 
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams.update({"font.size":22})


def get_recall(database_vectors, query_vectors):
    # Original PointNetVLAD code
    database_output = database_vectors
    queries_output = query_vectors
    # When embeddings are normalized, using Euclidean distance gives the same
    # nearest neighbour search results as using cosine distance
    database_nbrs = KDTree(database_output)
    threshold = max(int(round(len(database_output)/100.0)), 1)
    num_neighbors = threshold * 2
    num_evaluated = 0
    rank = []
    for i in range(len(queries_output)):
        # i is query element ndx
        true_neighbors = i
        num_evaluated += 1
        distances, indices = database_nbrs.query(np.array([queries_output[i]]), k=num_neighbors)
        rank.append(indices[0][0:5])
    return rank


def get_recall_bsd(database_vectors, query_vectors):
    # Original PointNetVLAD code
    database_output = database_vectors
    queries_output = query_vectors
    threshold = max(int(round(len(database_output)/100.0)), 1)
    num_neighbors = threshold * 2
    num_evaluated = 0
    rank = []
    for i in range(len(queries_output)):
        # i is query element ndx
        true_neighbors = i
        num_evaluated += 1
        distances = cdist(np.array([queries_output[i]]), database_output, metric='hamming')
        sorted_indices = np.argsort(distances)
        indices = sorted_indices[:, :num_neighbors]
        rank.append(indices[0][0:5])
    return rank


if __name__ == "__main__":
    location_name = 'hudsonriver5k'
    # MES
    model_name = '2d'
    embeddings = sio.loadmat(os.path.join('features_128D', location_name+'U_'+model_name+'.mat'))
    database_embeddings = embeddings['ref']
    query_embeddings = embeddings['qry']
    rank = get_recall(database_embeddings, query_embeddings) 
    np.save(os.path.join('ranks', location_name+'_'+model_name+'_rank.npy'), rank) # obtain rank for retrieving visualization

    model_name = '2dsafapolar'
    embeddings = sio.loadmat(os.path.join('features_128D', location_name+'U_'+model_name+'.mat'))
    database_embeddings = embeddings['ref']
    query_embeddings = embeddings['qry']
    rank = get_recall(database_embeddings, query_embeddings) 
    np.save(os.path.join('ranks', location_name+'_'+model_name+'_rank.npy'), rank) # obtain rank for retrieving visualization

    model_name = 'dgcnn2to3'
    embeddings = sio.loadmat(os.path.join('features_128D', location_name+'U_'+model_name+'.mat'))
    database_embeddings = embeddings['ref']
    query_embeddings = embeddings['qry']
    rank = get_recall(database_embeddings, query_embeddings) 
    np.save(os.path.join('ranks', location_name+'_'+model_name+'_rank.npy'), rank) # obtain rank for retrieving visualization

    # BSD, ES
    model_name = 'v2_12'
    embeddings = sio.loadmat(os.path.join('features_16D', 'ES_'+location_name+'_'+model_name+'.mat'))
    database_embeddings = embeddings['ref']
    query_embeddings = embeddings['qry']
    rank = get_recall(database_embeddings, query_embeddings) 
    np.save(os.path.join('ranks', location_name+'_'+model_name+'_rank.npy'), rank) # obtain rank for retrieving visualization

    model_name = 'resnet18'
    embeddings = sio.loadmat(os.path.join('features_16D', 'BSD_'+location_name+'_'+model_name+'.mat'))
    database_embeddings = embeddings['ref']
    query_embeddings = embeddings['qry']
    rank = get_recall_bsd(database_embeddings, query_embeddings) 
    np.save(os.path.join('ranks', location_name+'_'+model_name+'_rank.npy'), rank) # obtain rank for retrieving visualization
