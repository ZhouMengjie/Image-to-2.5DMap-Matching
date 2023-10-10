import os
os.environ["OMP_NUM_THREADS"] = '1'
import sys
sys.path.append(os.getcwd())

import numpy as np
import argparse
import torch
import scipy.io as sio
import random
import time
import torch.nn.functional as F

from sklearn.neighbors import KDTree
from config.utils import get_datetime
from models.get_model import get_model
from data.seq_dataset import SeqDataset
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA

def seed_all(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    print('Seed: {}'.format(random_seed))


def evaluate(model, device, params, exp_name, pca_dim):
    stats = {}
    datasets = {}
    dataloaders = {}
   
    nw = min([os.cpu_count(), params.batch_size if params.batch_size > 1 else 0, 8])  # number of workers
    model.eval()

    for location_name in params.eval_files:
        # Extract location name from query and database files
        datasets[location_name] = SeqDataset(params.dataset_folder, location_name, params.pre_model_name)
        dataloaders[location_name] = DataLoader(datasets[location_name], batch_size=params.val_batch_size, num_workers=nw, pin_memory=True, shuffle=False, drop_last=False)
        count = 0
        similarity = []
        one_percent_recall = []
        database_embeddings = []
        query_embeddings = []
        summary_time = 0
        summary_memory = 0
        for batch in dataloaders[location_name]:
            with torch.no_grad():
                batch = {e: batch[e].to(device) for e in batch}
                pre_pano = batch['pano']
                pre_map = batch['map']

                torch.cuda.synchronize()
                torch.cuda.reset_max_memory_allocated()
                start = time.time()
                if params.share:
                    pano_feat = model(pre_pano)
                    map_feat = model(pre_map)
                else:
                    pano_feat, map_feat = model(pre_pano, pre_map)
                torch.cuda.synchronize()
                end = time.time()
                memory = torch.cuda.max_memory_allocated(device=device)
                summary_time += (end-start)
                summary_memory += memory

                pano_embedding = torch.nn.functional.normalize(pano_feat, p=2, dim=1)  # Normalize embeddings, dim=row
                map_embedding = torch.nn.functional.normalize(map_feat, p=2, dim=1)
                pano_embedding = pano_embedding.detach().cpu().numpy()
                map_embedding = map_embedding.detach().cpu().numpy()

            torch.cuda.empty_cache()
            count = count + 1
            # check data here: batch idx
            print('Batch[{0}]/Location[{1}]: {2}'.format(count,location_name,batch['label']))
            
            database_embeddings.append(map_embedding)
            query_embeddings.append(pano_embedding)

        avg_time = (summary_time / count) / params.val_batch_size
        avg_memory = (summary_memory / count) / params.val_batch_size
        print("GPU inference time:", avg_time * 1000, "ms")
        print("GPU memory usage:", avg_memory / 1024**2, "MB")

        database_embeddings = np.vstack(database_embeddings)
        query_embeddings = np.vstack(query_embeddings)

        # PCA feature dimensionality reduction
        estimator = PCA(n_components=pca_dim)
        pca_database_embeddings = estimator.fit_transform(database_embeddings) 
        pca_query_embeddings = estimator.transform(query_embeddings)
        
        recall, similarity, one_percent_recall = get_recall(pca_database_embeddings, pca_query_embeddings)   
        ave_recall = recall
        average_similarity = np.mean(similarity)
        ave_one_percent_recall = one_percent_recall
        stats[location_name] = {'ave_one_percent_recall': ave_one_percent_recall, 'ave_recall': ave_recall,
                'average_similarity': average_similarity} 
        
        # obtain recall to plot figures
        # res = {}
        # res['recall'] = recall
        # model_name = os.path.split(params.weights)[1]
        # model_name = model_name.split('.')[0]
        # np.save(os.path.join('results', location_name+'_'+model_name+'_'+exp_name+'.npy'), recall)

    return stats


# def get_recall(database_vectors, query_vectors, location_name):
def get_recall(database_vectors, query_vectors):
    # Original PointNetVLAD code
    database_output = database_vectors
    queries_output = query_vectors

    # When embeddings are normalized, using Euclidean distance gives the same
    # nearest neighbour search results as using cosine distance
    database_nbrs = KDTree(database_output)

    top1_similarity_score = []
    threshold = max(int(round(len(database_output)/100.0)), 1)

    num_neighbors = threshold * 2
    recall = [0] * num_neighbors

    num_evaluated = 0
    # rank = []
    for i in range(len(queries_output)):
        # i is query element ndx
        true_neighbors = i
        num_evaluated += 1
        distances, indices = database_nbrs.query(np.array([queries_output[i]]), k=num_neighbors)
        # rank.append(indices[0][0:5])
        for j in range(len(indices[0])):
            if indices[0][j] == true_neighbors:
                if j == 0:
                    similarity = np.dot(queries_output[i], database_output[indices[0][j]])
                    top1_similarity_score.append(similarity)
                recall[j] += 1
                break
    # np.save(os.path.join('results', location_name+'_rank.npy'), rank) # obtain rank for retrieving visualization
    recall = (np.cumsum(recall)/float(num_evaluated))*100
    one_percent_recall = recall[threshold]
    return recall, top1_similarity_score, one_percent_recall


def print_eval_stats(stats):
    for database_name in stats:
        print('Dataset: {}'.format(database_name))
        t = 'Avg. top 1% recall: {:.2f}   Avg. similarity: {:.4f}   Avg. recall @N:'
        print(t.format(stats[database_name]['ave_one_percent_recall'], stats[database_name]['average_similarity']))
        print(stats[database_name]['ave_recall'])


def export_eval_stats(file_name, prefix, eval_stats):
    s = prefix
    ave_1p_recall_l = []
    ave_recall_l = []
    # Print results on the final model
    with open(file_name, "a") as f:
        for ds in params.eval_files:
            if ds not in eval_stats:
                continue
            ave_1p_recall = eval_stats[ds]['ave_one_percent_recall']
            ave_1p_recall_l.append(ave_1p_recall)
            ave_recall = eval_stats[ds]['ave_recall'][0]
            ave_recall_l.append(ave_recall)
            s += ", {:0.2f}, {:0.2f}".format(ave_1p_recall, ave_recall)

        mean_1p_recall = np.mean(ave_1p_recall_l)
        mean_recall = np.mean(ave_recall_l)
        s += ", {:0.2f}, {:0.2f}\n".format(mean_1p_recall, mean_recall)
        f.write(s)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Minkowski Net embeddings using BatchHard negative mining')
    parser.add_argument('--dataset_folder', type=str, default='datasets', required=False, help='Dataset folder')
    parser.add_argument('--batch_size', type=int, default=32, required=False, help='Training batch size')
    parser.add_argument('--val_batch_size', type=int, default=24, required=False, help='Testing batch size')
    parser.add_argument('--weights', type=str, required=False, help='Trained model weights')
    parser.add_argument('--epoch', type=int, default=1, required=False, help='Initial training epoch')    
    parser.add_argument('--eval_files', type=str, required=False, help='Eval files')
    parser.add_argument('--seed', type=int, default=1, required=False, help='Seed')
    parser.add_argument('--feat_dim',type=int, default=4096, required=False, help='Feature dimension')
    parser.add_argument('--pre_model_name', type=str, required=False, help='Precomputed model name')
    parser.add_argument('--num_layers', type=int, default=6, required=False, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8, required=False, help='Number of transformer heads')
    parser.add_argument('--seq_len', type=int, default=5, required=False, help='Sequence length')      
    parser.add_argument('--device', type=str, required=False, help='Device')  
    parser.add_argument('--exp_name', type=str, required=False, help='Experiment name')  
    parser.add_argument('--pca_dim', type=int, default=80, required=False, help='PCA dimension')  
    parser.add_argument('--model_type', type=str, required=False, help='Model type')
    parser.add_argument('--share', dest='share', action='store_true')

    parser.set_defaults(eval_files='hudsonriver5kU,unionsquare5kU,wallstreet5kU')
    parser.set_defaults(pre_model_name='resnetsafa_dgcnn_asam_2to3_up')
    parser.set_defaults(device='cuda')  
    parser.set_defaults(model_type='seqnet') 

    params = parser.parse_args()
    seed_all(params.seed)
    params.eval_files = [e for e in params.eval_files.split(',')]

    savedStdout = sys.stdout
    s = get_datetime()
    if not os.path.exists('test_logs'):
        os.mkdir('test_logs')
    print_log = open(os.path.join('test_logs',s+'.txt'),'w')
    sys.stdout = print_log

    if params.weights is None:
        w = 'RANDOM WEIGHTS'
    else:
        w = params.weights
    print('Weights: {}'.format(w))
    print('')

    print('Parameters:')
    param_dict = vars(params)
    for e in param_dict:
        print('{}: {}'.format(e, param_dict[e]))
    print('')

    device = params.device
    print('Device: {}'.format(device))
    print('PCA dim:{}'.format(params.pca_dim))

    model = get_model(params)
    model.to(device)
    model_size = torch.cuda.memory_allocated(device=device)
    print('Model size: ', model_size / 1024**2, 'MB')

    if params.weights is not None:
        assert os.path.exists(params.weights), 'Cannot open network weights: {}'.format(params.weights)
        checkpoint = torch.load(params.weights, map_location=device) 
        if 'model' in checkpoint:
            ckp = checkpoint['model']
        else:
            ckp = checkpoint
        state_dict = {}
        for k, v in ckp.items():
            new_k = k.replace('module.', '') if 'module' in k else k
            state_dict[new_k] = v
        model.load_state_dict(state_dict, strict=True) 
        print('load pretrained {} model!'.format(params.weights))

    stats = evaluate(model, device, params, params.exp_name, params.pca_dim)  
    print_eval_stats(stats)
    
    # Append key experimental metrics to experiment summary file
    prefix = "{}, {}".format(params.pre_model_name, w)
    export_eval_stats("experiment_results.txt", prefix, stats)

