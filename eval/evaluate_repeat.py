# Author: Jacek Komorowski
# Warsaw University of Technology

# Evaluation code adapted from PointNetVlad code: https://github.com/mikacuy/pointnetvlad
import os
os.environ["OMP_NUM_THREADS"] = '1'
import sys
sys.path.append(os.getcwd())

import numpy as np
import argparse
import torch
import scipy.io as sio
import random

from sklearn.neighbors import KDTree
from config.utils import MinkLocParams
from models.model_factory_tri import model_factory
from torch.utils.data import DataLoader
from data.augmentation import ValTransform, ValRGBTransform, ValTileTransform 
from data.streetlearn_pickle import StreetLearnDataset
from data.dataset_utils_pickle import make_collate_fn_torch

DEBUG = False

def seed_all(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    print('Seed: {}'.format(random_seed))


def evaluate(model, device, params, seed):
    stats = {}
    datasets = {}
    dataloaders = {}
    cloud_transform = ValTransform()
    img_transform = ValRGBTransform()
    tile_transform = ValTileTransform(params.model_params.tile_size)

    nw = min([os.cpu_count(), params.batch_size if params.batch_size > 1 else 0, 8])  # number of workers

    model.eval()

    runnnng_times = 1
    location_name = params.eval_files[0]
    ave_recall = 0
    average_similarity = 0
    ave_one_percent_recall = 0

    for i in range(runnnng_times):
        # Extract location name from query and database files
        datasets[location_name] = StreetLearnDataset(params.dataset_folder, location_name,
                                                    transform=cloud_transform,
                                                    image_size=params.model_params.img_size, image_transform=img_transform,
                                                    tile_size=params.model_params.tile_size, tile_transform=tile_transform,
                                                    use_cloud=params.use_cloud, use_rgb=params.use_rgb, use_tile=params.use_tile, 
                                                    use_feat=params.use_feat, use_polar=params.use_polar, 
                                                    normalize=params.pc_normalize, npoints=params.npoints)

        # datasets[location_name].transform.aug_mode = 0 
        collate_fn = make_collate_fn_torch(datasets[location_name])
        dataloaders[location_name] = DataLoader(datasets[location_name], batch_size=params.val_batch_size, collate_fn=collate_fn,
                                       num_workers=nw, pin_memory=True, shuffle=False, drop_last=False)

        count = 0
        similarity = []
        one_percent_recall = []

        database_embeddings = []
        query_embeddings = []

        for batch in dataloaders[location_name]:
            with torch.no_grad():
                batch = {e: batch[e].to(device) for e in batch}
                x = model(batch)

                cloud_embedding = x['embedding']
                image_embedding = x['image_embedding']
                if params.normalize_embeddings:
                    cloud_embedding = torch.nn.functional.normalize(cloud_embedding, p=2, dim=1)  # Normalize embeddings, dim=row
                    image_embedding = torch.nn.functional.normalize(image_embedding, p=2, dim=1)

                cloud_embedding = cloud_embedding.detach().cpu().numpy()
                image_embedding = image_embedding.detach().cpu().numpy()

            torch.cuda.empty_cache()
            count = count + 1
            # check data here: batch idx
            print('Batch[{0}]/Location[{1}]: {2}'.format(count,location_name,batch['labels']))
            
            database_embeddings.append(cloud_embedding)
            query_embeddings.append(image_embedding)

        database_embeddings = np.vstack(database_embeddings)
        query_embeddings = np.vstack(query_embeddings)

        # pred = {}
        # pred['ref'] = database_embeddings
        # pred['qry'] = query_embeddings
        # model_name = os.path.split(params.load_weights)[1]
        # model_name = model_name.split('.')[0]
        # sio.savemat(os.path.join('results', location_name+'_'+model_name+'_'+device+'.mat'), pred)

        recall, similarity, one_percent_recall = get_recall(database_embeddings, query_embeddings)   
        ave_recall += recall / runnnng_times
        average_similarity += np.mean(similarity) / runnnng_times
        ave_one_percent_recall += one_percent_recall / runnnng_times
        print('top 1 recall: {recall:.4f}\t'
                'top 1p recall: {one_percent_recall:.4f}\t'
                'similarity: {similarity:.4f}'.format(
                recall=recall[0], 
                one_percent_recall=one_percent_recall, similarity=np.mean(similarity)))

    stats[location_name] = {'ave_one_percent_recall': ave_one_percent_recall, 'ave_recall': ave_recall,
            'average_similarity': average_similarity} 
    return stats


def get_recall(database_vectors, query_vectors):
    # Original PointNetVLAD code
    database_output = database_vectors
    queries_output = query_vectors
    # indexes = np.random.choice(len(query_vectors), 5000, replace=False)
    # queries_output = query_vectors[indexes]
    # database_output = database_vectors[indexes]

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
    parser.add_argument('--config', type=str, required=False, help='Path to configuration file')
    parser.add_argument('--model_config', type=str, required=False, help='Path to the model-specific configuration file')    
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--batch_size', type=int, default=32, required=False, help='Training batch size')
    parser.add_argument('--weights', type=str, required=False, help='Trained model weights')
    parser.add_argument('--epoch', type=int, default=1, required=False, help='Initial training epoch')    
    parser.add_argument('--pc_normalize', dest='pc_normalize', action='store_true')
    parser.add_argument('--train_file', type=str, required=False, help='Train file')
    parser.add_argument('--val_file', type=str, required=False, help='Val file')
    parser.add_argument('--eval_files', type=str, required=False, help='Eval files')
    parser.add_argument('--distributed', dest='distributed', action='store_true')
    parser.add_argument('--val_distributed', dest='val_distributed', action='store_true')
    parser.add_argument('--freeze', dest='freeze', action='store_true')
    parser.add_argument('--syncBN', dest='syncBN', action='store_true')
    parser.add_argument('--seed', type=int, default=1, required=False, help='Seed')
    parser.add_argument('--port', type=str, default='12363', required=False, help='Port')  
    parser.add_argument('--mink_quantization_size', type=float, default=0.01, required=False, help='Quantization size')
    parser.add_argument('--model3d', type=str, required=False, help='Model 3D')  
    parser.add_argument('--use_feat', dest='use_feat', action='store_true')
    parser.add_argument('--use_polar', dest='use_polar', action='store_true')    
    parser.add_argument('--feat_dim',type=int, required=False, default=128, help='Feature dimension')
    parser.add_argument('--loss', type=str, required=False, help='Loss')
    parser.add_argument('--margin', type=float, default=0.1, required=False, help='Loss margin')
    parser.add_argument('--npoints', type=int, default=4096, required=False, help='Number of points')
    parser.add_argument('--nneighbor', type=int, default=8, required=False, help='Number of neighbors')    
    parser.add_argument('--fuse', type=str, required=False, help='Feature fusion method')
    parser.add_argument('--device', type=str, required=False, help='Device')  
    parser.set_defaults(config='config/config_train.txt')
    parser.set_defaults(model_config='models/minklocmultimodal.txt')
    parser.set_defaults(train_file='trainstreetlearnU_cmu5kU')
    parser.set_defaults(val_file='hudsonriver5kU')
    parser.set_defaults(eval_files='hudsonriver5kU')
    parser.set_defaults(model3d='pointnet')
    parser.set_defaults(loss='MultiInfoNCELoss')
    parser.set_defaults(fuse='concat')
    parser.set_defaults(device='cuda:0')
    

    args = parser.parse_args()
    seed_all(args.seed)

    print('Config path: {}'.format(args.config))
    print('Model config path: {}'.format(args.model_config))
    if args.weights is None:
        w = 'RANDOM WEIGHTS'
    else:
        w = args.weights
    print('Weights: {}'.format(w))
    print('')

    params = MinkLocParams(args)
    params.print()

    device = args.device
    # device = torch.device('cuda:0')
    print('Device: {}'.format(device))

    model = model_factory(params)
    model.to(device)

    if args.weights is not None:
        assert os.path.exists(params.load_weights), 'Cannot open network weights: {}'.format(params.load_weights)
        checkpoint = torch.load(params.load_weights, map_location=device)     
        state_dict = {}
        try:
            ckp = checkpoint['model']
            op_ckp = checkpoint['optimizer']
        except:
            ckp = checkpoint
            op_ckp = None
        for k, v in ckp.items():
            new_k = k.replace('module.', '') if 'module' in k else k
            if new_k == 'cloud_fe.backbone.conv0.kernel' and v.shape[1] != params.feat_size:
                new_k = 'ignore' 
            state_dict[new_k] = v
        model.load_state_dict(state_dict, strict=True) 
        print('load pretrained {} model!'.format(params.load_weights))

    stats = evaluate(model, device, params, args.seed)  
    print_eval_stats(stats)
    
    # Append key experimental metrics to experiment summary file
    model_params_name = os.path.split(params.model_params.model_params_path)[1]
    config_name = os.path.split(params.params_path)[1]
    model_name = os.path.split(params.load_weights)[1]
    prefix = "{}, {}, {}".format(model_params_name, config_name, model_name)
    export_eval_stats("experiment_results.txt", prefix, stats)

