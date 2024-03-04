import os
import sys
sys.path.append(os.getcwd())
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
plt.rc('font',family='Times New Roman') 
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams.update({"font.size":22})
# from sklearn.neighbors import KDTree
from scipy.spatial.distance import cdist


def get_recall(database_vectors, query_vectors):
    # Original PointNetVLAD code
    database_output = database_vectors
    queries_output = query_vectors

    threshold = max(int(round(len(database_output)/100.0)), 1)

    num_neighbors = threshold * 2
    recall = [0] * num_neighbors

    num_evaluated = 0
    for i in range(len(queries_output)):
        # i is query element ndx
        true_neighbors = i
        num_evaluated += 1
        distances = cdist(np.array([queries_output[i]]), database_output, metric='hamming')
        sorted_indices = np.argsort(distances)
        indices = sorted_indices[:, :num_neighbors]
        for j in range(len(indices[0])):
            if indices[0][j] == true_neighbors:
                recall[j] += 1
                break
    recall = (np.cumsum(recall)/float(num_evaluated))
    one_percent_recall = recall[threshold]
    return recall, one_percent_recall

if __name__ == '__main__':
    # load features and get recall
    # BSD_hudsonriver5k_resnet18
    mat = sio.loadmat(os.path.join('results','exps','BSD_hudsonriver5k_resnet18.mat'))
    database_embeddings = mat['ref']
    query_embeddings = mat['qry']
    recall_hr, recall_hr_op = get_recall(database_embeddings, query_embeddings) 

    mat = sio.loadmat(os.path.join('results','exps','BSD_unionsquare5k_resnet18.mat'))
    database_embeddings = mat['ref']
    query_embeddings = mat['qry']
    recall_us, recall_us_op = get_recall(database_embeddings, query_embeddings) 

    mat = sio.loadmat(os.path.join('results','exps','BSD_wallstreet5k_resnet18.mat'))
    database_embeddings = mat['ref']
    query_embeddings = mat['qry']
    recall_ws, recall_ws_op = get_recall(database_embeddings, query_embeddings) 

    # plot recall
    x = np.arange(1, len(recall_hr)+1) / 50
    plt.figure()
    plt.plot(x,recall_hr,color='blue',linewidth=2,linestyle='solid',label='HR ({:.1f}%)'.format(recall_hr_op*100))
    plt.plot(x,recall_us,color='red',linewidth=2,linestyle='solid',label='US ({:.1f}%)'.format(recall_us_op*100))
    plt.plot(x,recall_us,color='yellow',linewidth=2,linestyle='solid',label='WS ({:.1f}%)'.format(recall_ws_op*100))
    
    plt.xlabel('k (% of the dataset)')
    plt.xticks(np.arange(0, 2.5, step=0.5))
    plt.ylabel('Top-k% recall')
    plt.yticks(np.arange(0.0, 1.0, step=0.1))
    plt.ylim(0.0,1.0)
    plt.legend(loc=4)
    plt.grid(linestyle='solid', linewidth=0.5)
    # plt.savefig(os.path.join('results','chapter04',location_name+'_method.pdf'),bbox_inches='tight')
    plt.show()