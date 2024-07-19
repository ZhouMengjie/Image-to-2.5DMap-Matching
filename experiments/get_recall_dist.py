import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman') 
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams.update({"font.size":22})


def get_recall(database_vectors, query_vectors, locations):
    database_output = database_vectors
    queries_output = query_vectors
    database_nbrs = KDTree(database_output)
    thresholds = range(0, 51, 1)
    recall = [0] * len(thresholds)
    num_evaluated = 0
    loc2 = np.sum(locations ** 2, -1) 

    error_success = 0
    error_all = 0
    for i in range(len(queries_output)): # i-ground truth
        # i is query element ndx
        num_evaluated += 1
        _, pred_idx = database_nbrs.query(np.array([queries_output[i]]), k=1) # top1
        dist = -2 * np.matmul(locations, locations[i])
        dist += loc2
        dist +=  np.sum(locations[i] ** 2, -1)
        dist = np.sqrt(dist)
        error_all += dist[pred_idx[0][0]]
        for j in range(len(thresholds)):
            if dist[pred_idx[0][0]] <= thresholds[j]:
                recall[j] += 1  
                if j <= 10:  # <= 10m
                    error_success += dist[pred_idx[0][0]]
                break  
    print(error_all/float(num_evaluated))
    print(np.cumsum(recall)[10]/float(num_evaluated)*100)
    print(error_success/np.cumsum(recall)[10])
    recall = (np.cumsum(recall)/float(num_evaluated))*100
    return recall



if __name__ == "__main__":
    location_name = 'hudsonriver5kU'
    # baseline
    # model_name = '2d'
    # embeddings = sio.loadmat(os.path.join('features_128D', location_name+'_'+model_name+'.mat'))
    # database_embeddings = embeddings['ref']
    # query_embeddings = embeddings['qry']
    # db = pd.read_csv(os.path.join('datasets', 'csv', (location_name + '_xy.csv')), sep=',', header=None).values
    # x = np.expand_dims(pd.to_numeric(db[:,1]),axis=1)
    # y = np.expand_dims(pd.to_numeric(db[:,2]),axis=1)
    # locations = np.concatenate((x,y),-1)
    # recall_es = get_recall(database_embeddings, query_embeddings, locations) 
    # np.save(os.path.join('dist_results', location_name+'_'+model_name+'.npy'), recall_es)  

    # ours
    model_name = '2dsafapolar'
    embeddings = sio.loadmat(os.path.join('features_128D', location_name+'_'+model_name+'.mat'))
    database_embeddings = embeddings['ref']
    query_embeddings = embeddings['qry']
    db = pd.read_csv(os.path.join('datasets', 'csv', (location_name + '_xy.csv')), sep=',', header=None).values
    x = np.expand_dims(pd.to_numeric(db[:,1]),axis=1)
    y = np.expand_dims(pd.to_numeric(db[:,2]),axis=1)
    locations = np.concatenate((x,y),-1)
    recall_ours = get_recall(database_embeddings, query_embeddings, locations)  
    # np.save(os.path.join('dist_results', location_name+'_'+model_name+'.npy'), recall_ours)

    # plot
    # location_name = 'unionsquare5kU'
    # model_name = '2d'
    # recall_es = np.load(os.path.join('dist_results', location_name+'_'+model_name+'.npy'))
    # model_name = 'dgcnn2to3'
    # recall_ours = np.load(os.path.join('dist_results', location_name+'_'+model_name+'.npy'))
    
    # x = range(0, 51, 1)
    # markevery = 10
    # # display
    # plt.figure()
    # plt.plot(x,recall_es,color='red',linewidth=2,marker='*',markersize=5,markevery=markevery,
    #         linestyle='dashed',label='ES - {:.2f}%'.format(recall_es[10]))
    # plt.plot(x,recall_ours,color='blue',linewidth=2,marker='o',markersize=5,markevery=markevery,
    #         linestyle='solid',label='Ours - {:.2f}%'.format(recall_ours[10]))
    # plt.xlabel('distance threshold (m)')
    # plt.xticks(np.arange(0, 51, step=10))
    # plt.xlim(0,50)
    # plt.ylabel('Accuracy')
    # plt.yticks(np.arange(40, 101, step=10))
    # plt.ylim(40,101)
    # plt.legend(loc=4,fontsize='small')
    # plt.grid(linestyle='dashed', linewidth=0.5)
    # plt.savefig(os.path.join('dist_results','ch6',location_name+'_dist.pdf'),bbox_inches='tight')
    # plt.show()




