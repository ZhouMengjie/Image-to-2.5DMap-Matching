import os
import numpy as np
import scipy.io as sio
from sklearn.neighbors import KDTree

def get_recall(database_vectors, query_vectors):
    # Original PointNetVLAD code
    database_output = database_vectors
    queries_output = query_vectors
    # indexes = np.random.choice(len(query_vectors), 5000, replace=False)
    # print(indexes)
    # database_output = database_vectors[indexes]
    # queries_output = query_vectors[indexes]


    # When embeddings are normalized, using Euclidean distance gives the same
    # nearest neighbour search results as using cosine distance
    database_nbrs = KDTree(database_output)
    top1_similarity_score = []
    threshold = max(int(round(len(database_output)/100.0)), 1)
    num_neighbors = threshold * 2
    recall = [0] * num_neighbors

    num_evaluated = 0
    for i in range(len(queries_output)):
        # i is query element ndx
        true_neighbors = i
        num_evaluated += 1
        distances, indices = database_nbrs.query(np.array([queries_output[i]]), k=num_neighbors)

        for j in range(len(indices[0])):
            if indices[0][j] == true_neighbors:
                if j == 0:
                    similarity = np.dot(queries_output[i], database_output[indices[0][j]])
                    top1_similarity_score.append(similarity)
                recall[j] += 1
                break

    recall = (np.cumsum(recall)/float(num_evaluated))*100
    one_percent_recall = recall[threshold]
    return recall, top1_similarity_score, one_percent_recall


def print_eval_stats(stats):
    for database_name in stats:
        print('Dataset: {}'.format(database_name))
        t = 'Avg. top 1% recall: {:.2f}   Avg. similarity: {:.4f}   Avg. recall @N:'
        print(t.format(stats[database_name]['ave_one_percent_recall'], stats[database_name]['average_similarity']))
        print(stats[database_name]['ave_recall'])


if __name__ == "__main__":
    location_name = 'hudsonriver5kU' # trainstreetlearnU_minklocmultimodal_baseline_20220907_1338_latest_20220908_1500_latest_cuda
    model_name = 'minklocmultimodal_baseline_20221013_1633_latest_20221014_2100_latest_20221016_1557_latest'
    device = 'cuda'
    seed = 1
    np.random.seed(seed)
    embeddings = sio.loadmat(os.path.join('results', location_name+'_'+model_name+'_'+device+'_'+str(seed)+'.mat'))
    embeddings_aug = sio.loadmat(os.path.join('results', location_name+'_'+model_name+'_'+device+'_'+str(seed)+'_aug2.mat'))
    
    # database_embeddings = embeddings['ref']
    # query_embeddings = embeddings['qry']

    database_embeddings = embeddings['qry']
    query_embeddings = embeddings_aug['qry']

    stats = {}
    recall, similarity, one_percent_recall = get_recall(database_embeddings, query_embeddings)   
    ave_recall = recall
    average_similarity = np.mean(similarity)
    ave_one_percent_recall = one_percent_recall
    stats[location_name] = {'ave_one_percent_recall': ave_one_percent_recall, 'ave_recall': ave_recall,
                'average_similarity': average_similarity}
    print_eval_stats(stats)

