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

from sklearn.neighbors import KDTree
from config.utils import Params, get_datetime
from models.model_factory import model_factory
from torch.utils.data import DataLoader
from data.augmentation_simple import ValTransform, ValRGBTransform, ValTileTransform 
from data.streetlearn_pickle import StreetLearnDataset
from data.dataset_utils_pickle import make_collate_fn_torch
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
    cloud_transform = ValTransform()
    img_transform = ValRGBTransform()
    tile_transform = ValTileTransform(params.model_params.tile_size)

    nw = min([os.cpu_count(), params.batch_size if params.batch_size > 1 else 0, 8])  # number of workers
    model.eval()

    for location_name in params.eval_files:
        # Extract location name from query and database files
        datasets[location_name] = StreetLearnDataset(params.dataset_folder, location_name,
                                                    transform=cloud_transform,
                                                    image_size=params.model_params.img_size, image_transform=img_transform,
                                                    tile_size=params.model_params.tile_size, tile_transform=tile_transform,
                                                    use_cloud=params.use_cloud, use_rgb=params.use_rgb, use_tile=params.use_tile, 
                                                    use_feat=params.use_feat, use_polar=params.use_polar,
                                                    normalize=params.pc_normalize, npoints=params.npoints)

        collate_fn = make_collate_fn_torch(datasets[location_name])
        dataloaders[location_name] = DataLoader(datasets[location_name], batch_size=params.val_batch_size, collate_fn=collate_fn,
                                       num_workers=nw, pin_memory=True, shuffle=False, drop_last=False)

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
                torch.cuda.synchronize()
                torch.cuda.reset_max_memory_allocated()
                start = time.time()
                x = model(batch)
                torch.cuda.synchronize()
                end = time.time()
                memory = torch.cuda.max_memory_allocated(device=device)
                summary_time += (end-start)
                summary_memory += memory

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
        
        # obtain query and reference embeddings for subsequent localization tasks
        # pred = {}
        # pred['ref'] = pca_database_embeddings
        # pred['qry'] = pca_query_embeddings
        # model_name = os.path.split(params.load_weights)[1]
        # model_name = model_name.split('.')[0]
        # sio.savemat(os.path.join('results', location_name+'_'+model_name+'.mat'), pred)

        start_time = time.time()
        recall, similarity, one_percent_recall = get_recall(pca_database_embeddings, pca_query_embeddings) 
        # recall, similarity, one_percent_recall = get_recall(pca_database_embeddings, pca_query_embeddings, location_name)     
        end_time = time.time()
        run_time = (end_time-start_time) / 5000 * 1000
        print('average retrieving time is {}'.format(run_time))
        
        ave_recall = recall
        average_similarity = np.mean(similarity)
        ave_one_percent_recall = one_percent_recall
        stats[location_name] = {'ave_one_percent_recall': ave_one_percent_recall, 'ave_recall': ave_recall,
                'average_similarity': average_similarity} 
        
        # obtain recall to plot figures
        # res = {}
        # res['recall'] = recall
        # model_name = os.path.split(params.load_weights)[1]
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
    parser.add_argument('--config', type=str, required=False, help='Path to configuration file')
    parser.add_argument('--use_amp', dest='use_amp', action='store_true')
    parser.add_argument('--batch_size', type=int, default=32, required=False, help='Training batch size')
    parser.add_argument('--val_batch_size', type=int, default=24, required=False, help='Testing batch size')    
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
    parser.add_argument('--model3d', type=str, required=False, help='Model 3D')  
    parser.add_argument('--model2d_tile', type=str, required=False, help='Model 2D for Tile')  
    parser.add_argument('--model2d_pano', type=str, required=False, help='Model 2D for Pano')     
    parser.add_argument('--use_feat', dest='use_feat', action='store_true')
    parser.add_argument('--use_polar', dest='use_polar', action='store_true')
    parser.add_argument('--feat_dim',type=int, required=False, default=128, help='Feature dimension')
    parser.add_argument('--loss', type=str, required=False, help='Loss')
    parser.add_argument('--margin', type=float, default=0.1, required=False, help='Loss margin')
    parser.add_argument('--npoints', type=int, default=1024, required=False, help='Number of points')
    parser.add_argument('--nneighbor', type=int, default=20, required=False, help='Number of neighbors')
    parser.add_argument('--fuse', type=str, required=False, help='Feature fusion method')
    parser.add_argument('--fc', type=str, required=False, help='Final block')
    parser.add_argument('--optimizer', type=str, required=False, help='Optimizer')
    parser.add_argument('--wd', type=float, default=1e-4, required=False, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=60, required=False, help='Total training epochs')
    parser.add_argument('--lr', type=float, default=4e-5, required=False, help='Initial learning rate')
    parser.add_argument('--scheduler', type=str, required=False, help='LR Scheduler')
    parser.add_argument('--device', type=str, required=False, help='Device')  
    parser.add_argument('--exp_name', type=str, required=False, help='Experiment name')  
    parser.add_argument('--pca_dim', type=int, default=128, required=False, help='PCA dimension')  

    parser.set_defaults(config='config/config_train.txt')
    parser.set_defaults(train_file='trainstreetlearnU_cmu5kU')
    parser.set_defaults(val_file='hudsonriver5kU')
    parser.set_defaults(eval_files='hudsonriver5kU,unionsquare5kU,wallstreet5kU')
    parser.set_defaults(model3d='none')
    parser.set_defaults(model2d_tile='resnet_safa')
    parser.set_defaults(model2d_pano='resnet_safa')
    parser.set_defaults(loss='MultiInfoNCELoss')
    parser.set_defaults(fuse='concat')
    parser.set_defaults(optimizer='SAM')
    parser.set_defaults(device='cuda')   
    # parser.set_defaults(weights='weights/resnetsafa_polar_asam_simple.pth') 

    args = parser.parse_args()
    seed_all(args.seed)

    savedStdout = sys.stdout
    s = get_datetime()
    if not os.path.exists('test_logs'):
        os.mkdir('test_logs')
    print_log = open(os.path.join('test_logs',s+'.txt'),'w')
    sys.stdout = print_log

    print('Config path: {}'.format(args.config))
    if args.weights is None:
        w = 'RANDOM WEIGHTS'
    else:
        w = args.weights
    print('Weights: {}'.format(w))
    print('')

    params = Params(args)
    params.print()

    device = args.device
    print('Device: {}'.format(device))
    print('PCA dim:{}'.format(args.pca_dim))

    model = model_factory(params)
    model.to(device)
    model_size = torch.cuda.memory_allocated(device=device)
    print('Model size: ', model_size / 1024**2, 'MB')

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

    stats = evaluate(model, device, params, args.exp_name, args.pca_dim)  
    print_eval_stats(stats)
    
    # Append key experimental metrics to experiment summary file
    config_name = os.path.split(params.params_path)[1]
    model_name = os.path.split(params.load_weights)[1]
    prefix = "{}, {}".format(config_name, model_name)
    export_eval_stats("experiment_results.txt", prefix, stats)

