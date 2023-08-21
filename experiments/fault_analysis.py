import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.neighbors import KDTree
from experiments.stdata import STData
import torchvision.transforms as transforms
from data.augmentation_pc import RandomRotation, RandomCenterCrop
import yaml
from experiments.visualizer import visualize_pcd  

def visualize(coords):
    with open('color_map.yaml','r') as f:
        colors = yaml.load(f, Loader=yaml.FullLoader)  
    coord = []
    color = []
    for i in range(len(coords)):
        coord.append(coords[i][:3])
        class_id = np.argmax(coords[i][3:])
        color.append(np.divide(colors['color_map'][class_id], 255))
    coord = np.asarray(coord)
    color = np.asarray(color)
    visualize_pcd(coord, color, 'npy')

def get_recall(database_vectors, query_vectors, locations):
    # Original PointNetVLAD code
    database_output = database_vectors
    queries_output = query_vectors
    # indexes = np.random.choice(len(query_vectors), 5000, replace=False)
    # queries_output = query_vectors[indexes]
    # When embeddings are normalized, using Euclidean distance gives the same
    # nearest neighbour search results as using cosine distance
    database_nbrs = KDTree(database_output)
    top1_similarity_score = []
    threshold = max(int(round(len(database_output)/100.0)), 1)
    num_neighbors = threshold * 2
    recall = [0] * num_neighbors
    num_evaluated = 0
    loc2 = np.sum(locations ** 2, -1) 

    # for visualization
    t = [RandomRotation(axis=np.array([0, 1, 0])), RandomCenterCrop(radius=76)]
    cloud_transform = transforms.Compose(t)
    t = [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    img_transform = transforms.Compose(t)
    data = STData('datasets', 'unionsquare5kU', transform=cloud_transform, image_transform=img_transform)   

    for i in range(len(queries_output)):
        # i is query element ndx
        num_evaluated += 1
        distances, indices = database_nbrs.query(np.array([queries_output[i]]), k=num_neighbors)
        dist = -2 * np.matmul(locations, locations[i])
        dist += loc2
        dist +=  np.sum(locations[i] ** 2, -1)
        dist = np.sqrt(dist)

        for j in range(len(indices[0])):
            if dist[indices[0][j]] <= 25:
                if j == 0:
                    similarity = np.dot(queries_output[i], database_output[indices[0][j]])
                    top1_similarity_score.append(similarity)
                recall[j] += 1
                break     

        print(dist[indices[0][:4]])
        # fault analysis
        if (i==10 or i==100 or i==1000) and j > 5:
            anchor = data(i)
            coords = anchor['cloud'].numpy()
            visualize(coords)
            for k in range(4):
                # check top 5 fault analysis, visualization
                error_case = data(indices[0][k])
                coords = error_case['cloud'].numpy()
                visualize(coords)

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
    location_name = 'unionsquare5kU' # trainstreetlearnU_minklocmultimodal_baseline_20220907_1338_latest_20220908_1500_latest_cuda
    model_name = 'minklocmultimodal_baseline_20220920_1539_latest_20220921_1751_latest'
    device = 'cuda'
    embeddings = sio.loadmat(os.path.join('results', location_name+'_'+model_name+'_'+device+'.mat'))
    database_embeddings = embeddings['ref']
    query_embeddings = embeddings['qry']

    db = pd.read_csv(os.path.join('datasets', 'csv', (location_name + '_xy.csv')), sep=',', header=None).values
    x = np.expand_dims(pd.to_numeric(db[:,1]),axis=1)
    y = np.expand_dims(pd.to_numeric(db[:,2]),axis=1)
    locations = np.concatenate((x,y),-1)

    stats = {}
    recall, similarity, one_percent_recall = get_recall(database_embeddings, query_embeddings, locations)   
    ave_recall = recall
    average_similarity = np.mean(similarity)
    ave_one_percent_recall = one_percent_recall
    stats[location_name] = {'ave_one_percent_recall': ave_one_percent_recall, 'ave_recall': ave_recall,
            'average_similarity': average_similarity}
    print_eval_stats(stats)

