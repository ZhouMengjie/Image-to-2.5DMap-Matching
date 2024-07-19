import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman') 
# plt.rcParams["font.weight"] = "bold"
# plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams.update({"font.size":22})


def get_recall(database_vectors, query_vectors, locations):
    database_output = database_vectors
    queries_output = query_vectors
    database_nbrs = KDTree(database_output)
    thresholds = range(0, 51, 1)
    recall = [0] * len(thresholds)
    num_evaluated = 0
    loc2 = np.sum(locations ** 2, -1) 
    num_neighbors = max(int(round(len(database_output)/100.0)), 1)
    error = 0
    for i in range(len(queries_output)): # i-ground truth
        # i is query element ndx
        num_evaluated += 1
        _, indices = database_nbrs.query(np.array([queries_output[i]]), k=num_neighbors) # top1
        dist = -2 * np.matmul(locations, locations[i])
        dist += loc2
        dist +=  np.sum(locations[i] ** 2, -1)
        dist = np.sqrt(dist)
        min_dist = min(dist[idx] for idx in indices[0])
        error += min_dist
        for j in range(len(thresholds)):
            if min_dist <= thresholds[j]:
                recall[j] += 1  
                break  

    recall = (np.cumsum(recall)/float(num_evaluated))
    error = error/float(num_evaluated)
    return recall, error




if __name__ == "__main__":
    model_name = 'v2_12'
    ds_type = 'ES'
    
    # hr
    location_name = 'hudsonriver5k'
    embeddings = sio.loadmat(os.path.join('features_16D', ds_type+'_'+location_name+'_'+model_name+'.mat'))
    database_embeddings = embeddings['ref']
    query_embeddings = embeddings['qry']
    db = pd.read_csv(os.path.join('datasets', 'csv', (location_name + 'U_xy.csv')), sep=',', header=None).values
    x = np.expand_dims(pd.to_numeric(db[:,1]),axis=1)
    y = np.expand_dims(pd.to_numeric(db[:,2]),axis=1)
    locations = np.concatenate((x,y),-1)
    recall_hr, error_hr = get_recall(database_embeddings, query_embeddings, locations) 
    np.save(os.path.join('dist_results', ds_type+'_'+location_name+'_'+model_name+'.npy'), recall_hr)  

    # # us
    location_name = 'unionsquare5k'
    embeddings = sio.loadmat(os.path.join('features_16D', ds_type+'_'+location_name+'_'+model_name+'.mat'))
    database_embeddings = embeddings['ref']
    query_embeddings = embeddings['qry']
    db = pd.read_csv(os.path.join('datasets', 'csv', (location_name + 'U_xy.csv')), sep=',', header=None).values
    x = np.expand_dims(pd.to_numeric(db[:,1]),axis=1)
    y = np.expand_dims(pd.to_numeric(db[:,2]),axis=1)
    locations = np.concatenate((x,y),-1)
    recall_us, error_us = get_recall(database_embeddings, query_embeddings, locations) 
    np.save(os.path.join('dist_results', ds_type+'_'+location_name+'_'+model_name+'.npy'), recall_us) 

    # # ws
    location_name = 'wallstreet5k'
    embeddings = sio.loadmat(os.path.join('features_16D', ds_type+'_'+location_name+'_'+model_name+'.mat'))
    database_embeddings = embeddings['ref']
    query_embeddings = embeddings['qry']
    db = pd.read_csv(os.path.join('datasets', 'csv', (location_name + 'U_xy.csv')), sep=',', header=None).values
    x = np.expand_dims(pd.to_numeric(db[:,1]),axis=1)
    y = np.expand_dims(pd.to_numeric(db[:,2]),axis=1)
    locations = np.concatenate((x,y),-1)
    recall_ws, error_ws = get_recall(database_embeddings, query_embeddings, locations) 
    np.save(os.path.join('dist_results', ds_type+'_'+location_name+'_'+model_name+'.npy'), recall_ws)  
 
    # plot
    # recall_hr = np.load(os.path.join('dist_results', ds_type+'_'+'hudsonriver5k'+'_'+model_name+'.npy'))
    # recall_us = np.load(os.path.join('dist_results', ds_type+'_'+'unionsquare5k'+'_'+model_name+'.npy'))
    # recall_ws = np.load(os.path.join('dist_results', ds_type+'_'+'wallstreet5k'+'_'+model_name+'.npy'))

    x = range(0, 51, 1)
    markevery = 10
    # display
    plt.figure()
    plt.plot(x,recall_hr,color='#1f77b4',linewidth=3,
        linestyle='solid',label='HR ({:.2f}%)'.format(error_hr))
    plt.plot(x,recall_us,color='#d62728',linewidth=3,
            linestyle='solid',label='US ({:.2f}%)'.format(recall_us[10]*100))
    plt.plot(x,recall_ws,color='#ff7f0e',linewidth=3,
            linestyle='solid',label='WS ({:.2f}%)'.format(recall_ws[10]*100))
    plt.xlabel('distance threshold (m)')
    plt.xticks(np.arange(0, 51, step=10))
    plt.xlim(0,50)
    plt.ylabel('Accuracy')
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.ylim(0.7,1.0)
    plt.legend(loc=4,fontsize='small')
    plt.grid(linestyle='dashed', linewidth=0.5)
    plt.savefig(os.path.join('dist_results',ds_type+'_dist.pdf'),bbox_inches='tight')
    plt.show()




